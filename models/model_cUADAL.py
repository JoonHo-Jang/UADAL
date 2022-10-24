from __future__ import print_function
import argparse
import time
import datetime
from utils import utils
from utils.utils import OptimWithSheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models.function import HLoss
from models.function import BetaMixture1D
from models.function import CrossEntropyLoss
from models.basenet import *
import copy
from utils.utils import inverseDecayScheduler, CosineScheduler, StepScheduler, ConstantScheduler

class cUADAL():
    def __init__(self, args, num_class, src_dset, target_dset):
        self.model = 'cUADAL'
        self.args = args
        self.all_num_class = num_class
        self.known_num_class = num_class - 1  # except unknown
        self.dataset = args.dataset
        self.src_dset = src_dset
        self.target_dset = target_dset

        self.device = self.args.device
        self.build_model_init()

        self.ent_criterion = HLoss()
        self.bmm_model = self.cont = self.k = 0
        self.bmm_model_maxLoss = torch.log(torch.FloatTensor([self.known_num_class])).to(self.device)
        self.bmm_model_minLoss = torch.FloatTensor([0.0]).to(self.device)
        self.bmm_update_cnt = 0
        self.bmm_update_cnt_test = 0

        self.src_train_loader, self.src_val_loader, self.src_test_loader, self.src_train_idx = src_dset.get_loaders(
            class_balance_train=True)
        self.target_train_loader, self.target_val_loader, self.target_test_loader, self.tgt_train_idx = target_dset.get_loaders()
        self.num_batches = min(len(self.src_train_loader), len(self.target_train_loader))

        self.flag_entropy = False
        self.cutoff = False
        if self.args.dataset.lower() == 'visda':
            self.cutoff = True


    def build_model_init(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)

        self.G, self.E, self.C = utils.get_model_init(self.args, known_num_class=self.known_num_class, all_num_class=self.all_num_class)
        if self.args.cuda:
            self.G.to(self.args.device)
            self.E.to(self.args.device)
            self.C.to(self.args.device)

        scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                  max_iter=self.args.warmup_iter)

        if 'vgg' == self.args.net:
            for name,param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_w_g = OptimWithSheduler(optim.SGD(params, lr=self.args.g_lr * self.args.e_lr, weight_decay=5e-4, momentum=0.9,
                               nesterov=True), scheduler)
        self.opt_w_e = OptimWithSheduler(optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_w_c = OptimWithSheduler(optim.SGD(self.C.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)


    def build_model(self):
        def weights_init_bias_zero(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
        feat_mul_dim = self.G.output_num()*self.all_num_class
        self.flag_random_layer = False
        if feat_mul_dim > 4096:
            self.flag_random_layer = True
        cls_dim = self.all_num_class
        if self.flag_random_layer:
            self.random_layer = RandomLayer([self.G.output_num(), cls_dim])
            self.args.dc_out_dim = self.random_layer.output_dim
        else:
            self.args.dc_out_dim = self.G.output_num() * cls_dim

        _, _, self.E, self.DC = utils.get_model(self.args, known_num_class=self.known_num_class,
                                                         all_num_class=self.all_num_class, domain_dim=3,
                                                         dc_out_dim=self.args.dc_out_dim)

        self.DC.apply(weights_init_bias_zero)

        if self.args.cuda:
            self.E.to(self.args.device)
            self.DC.to(self.args.device)
            if self.flag_random_layer:
                self.random_layer.to(self.args.device)
                self.random_layer.cuda()


        SCHEDULER = {'cos': CosineScheduler, 'step': StepScheduler, 'id': inverseDecayScheduler, 'constant':ConstantScheduler}
        scheduler = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter)
        scheduler_dc = lambda step, initial_lr: SCHEDULER[self.args.scheduler](step, initial_lr, gamma=10, power=0.75,
                                                                            max_iter=self.num_batches*self.args.training_iter*self.args.update_freq_D)

        if 'vgg' == self.args.net:
            for name,param in self.G.named_parameters():
                if 'lower' in name:
                    param.requires_grad = False
                elif 'upper' in name:
                    param.requires_grad = False
            params = list(list(self.G.linear1.parameters()) + list(self.G.linear2.parameters()) + list(
                self.G.bn1.parameters()) + list(self.G.bn2.parameters()))
        else:
            params = list(self.G.parameters())

        self.opt_g = OptimWithSheduler(
            optim.SGD(params, lr=self.args.g_lr* self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_c = OptimWithSheduler(
            optim.SGD(self.C.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler)
        self.opt_dc = OptimWithSheduler(
            optim.SGD(self.DC.parameters(), lr=self.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True), scheduler_dc)

        scheduler_e = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=0, power=0.75,
                                                                     max_iter=self.num_batches*self.args.training_iter)
        self.opt_e = OptimWithSheduler(
            optim.SGD(self.E.parameters(), lr=self.args.e_lr, weight_decay=5e-4, momentum=0.9, nesterov=True),
            scheduler_e)

    def network_initialization(self):
        if 'resnet' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()
        elif 'vgg' in self.args.net:
            try:
                self.E.fc.reset_parameters()
                self.E.bottleneck.reset_parameters()
            except:
                self.E.fc.reset_parameters()

    def train_init(self):
        print('train_init starts')
        t1 = time.time()
        epoch_cnt =0
        step=0
        while step < self.args.warmup_iter + 1:
            self.G.train()
            self.E.train()
            self.C.train()
            epoch_cnt +=1
            for batch_idx, ((_, _, img_s_aug), label_s, _) in enumerate(self.src_train_loader):
                if self.args.cuda:
                    img_s =img_s_aug[0]
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))

                step += 1
                if step >= self.args.warmup_iter + 1:
                    break

                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()
                feat_s = self.G(img_s)
                out_s = self.E(feat_s)

                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                loss_s = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_s, dim=1))

                out_Cs = self.C(feat_s)
                label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                label_s_onehot = label_s_onehot + self.args.ls_eps / (self.all_num_class)
                loss_Cs = CrossEntropyLoss(label=label_s_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                loss = loss_s + loss_Cs

                loss.backward()
                self.opt_w_g.step()
                self.opt_w_e.step()
                self.opt_w_c.step()
                self.opt_w_g.zero_grad()
                self.opt_w_e.zero_grad()
                self.opt_w_c.zero_grad()

        duration = str(datetime.timedelta(seconds=time.time() - t1))[:7]
        print('train_init end with duration: %s' % duration)


    def train(self):
        print('Train Starts')
        step = 0
        t1 = time.time()
        for epoch in range(1, self.args.training_iter):
            joint_loader = zip(self.src_train_loader, self.target_train_loader)
            alpha = float((float(2) / (1 + np.exp(-10 * float((float(epoch) / float(self.args.training_iter)))))) - 1)
            for batch_idx, (((img_s, _, _), label_s, _), ((img_t, img_t_og, img_t_aug), label_t, index_t)) in enumerate(joint_loader):
                self.G.train()
                self.C.train()
                self.DC.train()
                self.E.train()
                if self.args.cuda:
                    img_s = Variable(img_s.to(self.args.device))
                    label_s = Variable(label_s.to(self.args.device))
                    img_t = Variable(img_t.to(self.args.device))
                    img_t_og = Variable(img_t_og.to(self.args.device))
                    img_t_aug = Variable(img_t_aug[0].to(self.args.device))

                out_t_free = self.E_freezed(self.G_freezed(img_t_og)).detach()
                w_unk_posterior = self.compute_probabilities_batch(out_t_free, 1)
                w_k_posterior = 1 - w_unk_posterior
                w_k_posterior = w_k_posterior.to(self.args.device)
                w_unk_posterior = w_unk_posterior.to(self.args.device)

                label_ds = Variable(torch.zeros(img_s.size()[0], dtype=torch.long).to(self.args.device))
                label_ds = nn.functional.one_hot(label_ds, num_classes=3)
                label_dt_known = Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                label_dt_known = nn.functional.one_hot(label_dt_known, num_classes=3)
                label_dt_unknown = 2 * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                label_dt_unknown = nn.functional.one_hot(label_dt_unknown, num_classes=3)
                #########################################################################################################
                for d_step in range(self.args.update_freq_D):
                    self.opt_dc.zero_grad()
                    feat_s = self.G(img_s).detach()
                    feat_t = self.G(img_t).detach()

                    label_dt = w_k_posterior[:, None] * label_dt_known + w_unk_posterior[:, None] * label_dt_unknown

                    label_d = torch.cat((label_ds, label_dt), dim=0)
                    features = torch.cat((feat_s, feat_t), dim=0)

                    out_s = self.C(feat_s)
                    out_t = self.C(feat_t)
                    outputs = torch.cat((out_s, out_t), dim=0)
                    softmax_out = nn.Softmax(dim=1)(outputs)  # .detach()
                    softmax_output = softmax_out.detach()

                    if self.flag_random_layer:
                        random_out = self.random_layer.forward([features, softmax_output])
                        op_out2 = random_out.view(-1, random_out.size(1))
                    else:
                        op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
                        op_out2 = op_out.view(-1, softmax_output.size(1) * features.size(1))
                    out_d = self.DC(op_out2)

                    if self.flag_entropy:
                        entropy = self.Entropy(softmax_out)
                        entropy.register_hook(self.grl_hook(utils.calc_coeff(step)))
                        entropy = 1.0 + torch.exp(-entropy)
                        source_mask = torch.ones_like(entropy)
                        source_mask[self.args.batch_size // 2:] = 0
                        source_weight = entropy * source_mask
                        target_mask = torch.ones_like(entropy)
                        target_mask[0:self.args.batch_size // 2] = 0
                        target_weight = entropy * target_mask
                        weight = source_weight / torch.sum(source_weight).detach().item() + \
                                 target_weight / torch.sum(target_weight).detach().item()
                        loss_d = CrossEntropyLoss(label=label_d, predict_prob=F.softmax(out_d, dim=1),
                                                  instance_level_weight=weight)
                    else:
                        loss_d = CrossEntropyLoss(label=label_d, predict_prob=F.softmax(out_d, dim=1))

                    loss_D = 0.5 * loss_d
                    loss_D.backward()
                    if self.args.opt_clip >0.0:
                        torch.nn.utils.clip_grad_norm_(self.DC.parameters(), self.args.opt_clip)
                    self.opt_dc.step()
                    self.opt_dc.zero_grad()
                #########################################################################################################
                for _ in range(self.args.update_freq_G):
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

                    feat_s = self.G(img_s)
                    feat_t = self.G(img_t)

                    label_dt = w_k_posterior[:, None] * label_dt_known - w_unk_posterior[:, None] * label_dt_unknown
                    label_d = torch.cat((label_ds, label_dt), dim=0)
                    features = torch.cat((feat_s, feat_t), dim=0)

                    out_s = self.C(feat_s)
                    out_t = self.C(feat_t)
                    outputs = torch.cat((out_s, out_t), dim=0)
                    softmax_out = nn.Softmax(dim=1)(outputs)  # .detach()
                    softmax_output = softmax_out.detach()

                    if self.flag_random_layer:
                        random_out = self.random_layer.forward([features, softmax_output])
                        op_out2 = random_out.view(-1, random_out.size(1))
                    else:
                        op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
                        op_out2 = op_out.view(-1, softmax_output.size(1) * features.size(1))
                    out_d = self.DC(op_out2)

                    if self.flag_entropy:
                        entropy = self.Entropy(softmax_out)
                        entropy.register_hook(self.grl_hook(utils.calc_coeff(step)))
                        entropy = 1.0 + torch.exp(-entropy)
                        source_mask = torch.ones_like(entropy)
                        source_mask[self.args.batch_size // 2:] = 0
                        source_weight = entropy * source_mask
                        target_mask = torch.ones_like(entropy)
                        target_mask[0:self.args.batch_size // 2] = 0
                        target_weight = entropy * target_mask
                        weight = source_weight / torch.sum(source_weight).detach().item() + \
                                 target_weight / torch.sum(target_weight).detach().item()
                        loss_d = CrossEntropyLoss(label=label_d, predict_prob=F.softmax(out_d, dim=1),
                                                  instance_level_weight=weight)
                    else:
                        loss_d = CrossEntropyLoss(label=label_d, predict_prob=F.softmax(out_d, dim=1))
                    loss_G = alpha * (- loss_d)
                    out_Es = self.E(feat_s)
                    label_s_onehot = nn.functional.one_hot(label_s, num_classes=self.known_num_class)
                    label_s_onehot = label_s_onehot * (1 - self.args.ls_eps)
                    label_s_onehot = label_s_onehot + self.args.ls_eps / (self.known_num_class)
                    loss_cls_Es = CrossEntropyLoss(label=label_s_onehot,
                                                   predict_prob=F.softmax(out_Es, dim=1))

                    out_Cs = self.C(feat_s)
                    label_Cs_onehot = nn.functional.one_hot(label_s, num_classes=self.all_num_class)
                    label_Cs_onehot = label_Cs_onehot * (1 - self.args.ls_eps)
                    label_Cs_onehot = label_Cs_onehot + self.args.ls_eps / (self.all_num_class)
                    loss_cls_Cs = CrossEntropyLoss(label=label_Cs_onehot, predict_prob=F.softmax(out_Cs, dim=1))

                    label_unknown = (self.known_num_class) * Variable(torch.ones(img_t.size()[0], dtype=torch.long).to(self.args.device))
                    label_unknown = nn.functional.one_hot(label_unknown, num_classes=self.all_num_class)
                    label_unknown_lsr = label_unknown * (1 - self.args.ls_eps)
                    label_unknown_lsr = label_unknown_lsr + self.args.ls_eps / (self.all_num_class)

                    feat_t_aug = self.G(img_t_aug)
                    out_Ct = self.C(feat_t)
                    out_Ct_aug = self.C(feat_t_aug)
                    if self.cutoff:
                        w_unk_posterior[w_unk_posterior < self.args.threshold] = 0.0
                        w_k_posterior[w_k_posterior < self.args.threshold] = 0.0

                    loss_cls_Ctu = alpha*CrossEntropyLoss(label=label_unknown_lsr, predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=w_unk_posterior)

                    pseudo_label = torch.softmax(out_Ct.detach(), dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    targets_u_onehot = nn.functional.one_hot(targets_u, num_classes=self.all_num_class)
                    mask = max_probs.ge(self.args.threshold).float()
                    loss_ent_Ctk = CrossEntropyLoss(label=targets_u_onehot,
                                                    predict_prob=F.softmax(out_Ct_aug, dim=1),
                                                    instance_level_weight=mask)

                    loss = loss_cls_Es + loss_cls_Cs + 0.5*loss_G + 0.5 * loss_ent_Ctk + 0.2 * loss_cls_Ctu
                    loss.backward()
                    self.opt_g.step()
                    self.opt_c.step()
                    self.opt_e.step()
                    self.opt_g.zero_grad()
                    self.opt_c.zero_grad()
                    self.opt_e.zero_grad()

            if (epoch % self.args.update_term == 0):
                C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos = self.test(epoch)

                self.args.logger.info(
                    'Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_Time_{}_BMM{}'.format(
                        epoch, self.args.training_iter, C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos,
                        str(datetime.timedelta(seconds=time.time() - t1))[:7], self.bmm_update_cnt))
                t1 = time.time()

        C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos = self.test(self.args.training_iter)
        self.args.logger.info(
            'Epoch_{:>3}/{:>3}_OS_{:.3f}_OS*_{:.3f}_UNK_{:.3f}_HOS_{:.3f}_Time_{}_BMM{}'.format(
                self.args.training_iter, self.args.training_iter, C_acc_os, C_acc_os_star, C_acc_unknown, C_acc_hos,
                str(datetime.timedelta(seconds=time.time() - t1))[:7], self.bmm_update_cnt))

    def compute_probabilities_batch(self, out_t, unk=1):
        ent_t = self.ent_criterion(out_t)
        batch_ent_t = (ent_t - self.bmm_model_minLoss) / (self.bmm_model_maxLoss - self.bmm_model_minLoss + 1e-6)
        batch_ent_t[batch_ent_t >= 1] = 1.0
        batch_ent_t[batch_ent_t <= 0] = 0.0
        B = self.bmm_model.posterior(batch_ent_t.clone().cpu().numpy(), unk)
        B = torch.FloatTensor(B)
        return B

    def freeze_GE(self):
        self.G_freezed = copy.deepcopy(self.G)
        self.E_freezed = copy.deepcopy(self.E)

    def test(self, epoch):
        self.G.eval()
        self.C.eval()
        self.E.eval()
        total_pred_t = np.array([])
        total_label_t = np.array([])
        all_ent_t = torch.Tensor([])
        with torch.no_grad():
            for batch_idx, ((img_t, _, _), label_t, index_t) in enumerate(self.target_test_loader):
                if self.args.cuda:
                    img_t, label_t = Variable(img_t.to(self.args.device)), Variable(label_t.to(self.args.device))
                feat_t = self.G(img_t)
                out_t = F.softmax(self.C(feat_t), dim=1)

                pred = out_t.data.max(1)[1]
                pred_numpy = pred.cpu().numpy()
                total_pred_t = np.append(total_pred_t, pred_numpy)
                total_label_t = np.append(total_label_t, label_t.cpu().numpy())

                out_Et = self.E(feat_t)
                ent_Et = self.ent_criterion(out_Et)
                all_ent_t = torch.cat((all_ent_t, ent_Et.cpu()))


        max_target_label = int(np.max(total_label_t)+1)
        m = utils.extended_confusion_matrix(total_label_t, total_pred_t, true_labels=list(range(max_target_label)), pred_labels=list(range(self.all_num_class)))
        cm = m
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)
        acc_os_star = sum([cm[i][i] for i in range(self.known_num_class)]) / self.known_num_class
        acc_unknown = sum([cm[i][self.known_num_class] for i in range(self.known_num_class, int(np.max(total_label_t)+1))]) / (max_target_label - self.known_num_class)
        acc_os = (acc_os_star * (self.known_num_class) + acc_unknown) / (self.known_num_class+1)
        acc_hos = (2 * acc_os_star * acc_unknown) / (acc_os_star + acc_unknown)

        self.G.train()
        self.C.train()
        self.E.train()

        if epoch%self.args.update_term==0:
            entropy_list = all_ent_t.data.numpy()
            loss_tr_t = (entropy_list - self.bmm_model_minLoss.data.cpu().numpy()) / (
                    self.bmm_model_maxLoss.data.cpu().numpy() - self.bmm_model_minLoss.data.cpu().numpy() + 1e-6)
            loss_tr_t[loss_tr_t >= 1] = 1 - 10e-4
            loss_tr_t[loss_tr_t <= 0] = 10e-4
            self.bmm_model = BetaMixture1D()
            self.bmm_model.fit(loss_tr_t)
            self.bmm_model.create_lookup(1)
            self.bmm_update_cnt += 1
            self.freeze_GE()
            self.network_initialization()

        return acc_os, acc_os_star, acc_unknown, acc_hos

    def Entropy(self, input_):
        bs = input_.size(0)
        epsilon = 1e-5
        entropy = -input_ * torch.log(input_ + epsilon)
        entropy = torch.sum(entropy, dim=1)
        return entropy

    def grl_hook(self, coeff):
        def fun1(grad):
            return -coeff * grad.clone()
        return fun1
