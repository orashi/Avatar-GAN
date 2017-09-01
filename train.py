import argparse
import os
import random
from functools import reduce
from math import log10

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils
from visdom import Visdom

from data.aData import CreateDataLoader
from models.Res import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--cut', type=int, default=2, help='cut backup frequency')
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambd', type=float, default=10, help='lambda for dragan')
parser.add_argument('--gp', action='store_true', help='train with wgan-gp')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--manualSeed', type=int, default=2345, help='random seed to use. Default=1234')
parser.add_argument('--geni', type=int, default=0, help='continue gen image num')
parser.add_argument('--epoi', type=int, default=0, help='continue epoch num')
parser.add_argument('--env', type=str, default='main', help='visdom env')

opt = parser.parse_args()
print(opt)

####### regular set up
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
gen_iterations = opt.geni
lambda_ = opt.lambd
try:
    os.makedirs(opt.outf)
except OSError:
    pass
# random seed setup
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
cudnn.benchmark = True
####### regular set up end


viz = Visdom(env=opt.env)

dataloader_train = CreateDataLoader(opt)

netG = Gnet()
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = PatchD()
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
criterion_L2 = nn.MSELoss()
noise = torch.FloatTensor(opt.batchSize, 100, 1, 1)
fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize, 1, 1, 1)
real_label = 1
fake_label = 0

one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion_L1.cuda()
    criterion_L2.cuda()
    criterion_GAN.cuda()
    label = label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    one, mone = one.cuda(), mone.cuda()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.9))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.9))


# schedulerG = lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', verbose=True, min_lr=0.000005,
#                                             patience=10)  # 1.5*10^5 iter
# schedulerD = lr_scheduler.ReduceLROnPlateau(optimizerD, mode='max', verbose=True, min_lr=0.000005,
#                                             patience=10)  # 1.5*10^5 iter
def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(opt.batchSize, 1, 1, 1)
    # alpha = alpha.expand(opt.batchSize, real_data.nelement() / opt.batchSize).contiguous().view(opt.batchSize, 3, 64,
    #                                                                                             64)
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                         disc_interpolates.size()),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambd
    return gradient_penalty


flag = 1
flag2 = 1
flag3 = 1
flag4 = 1
flag5 = 1
flag6 = 1
for epoch in range(opt.epoi, opt.niter):

    data_iter = iter(dataloader_train)
    iter_count = 0

    while iter_count < len(dataloader_train):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():
            p.requires_grad = False  # to avoid computation

        # train the discriminator Diters times
        Diters = opt.Diters

        j = 0
        while j < Diters and iter_count < len(dataloader_train):

            j += 1
            netD.zero_grad()

            data = data_iter.next()
            iter_count += 1

            if opt.cuda:
                real_sim = data.cuda()

            if opt.gp:
                # train with fake
                noise.normal_(0, 1)
                fake_sim = netG(Variable(noise, volatile=True)).data

                errD_fake = netD(Variable(fake_sim)).mean(0).view(1)
                errD_fake.backward(one, retain_graph=True)  # backward on score on real

                errD_real = netD(Variable(real_sim)).mean(0).view(1)
                errD_real.backward(mone, retain_graph=True)  # backward on score on real

                errD = errD_real - errD_fake

                # gradient penalty
                gradient_penalty = calc_gradient_penalty(netD, real_sim, fake_sim)
                gradient_penalty.backward()
            else:
                # train with fake
                labelv = Variable(label.fill_(fake_label))
                noise.normal_(0, 1)
                fake_sim = netG(Variable(noise, volatile=True)).data

                errD_fake = criterion_GAN(netD(Variable(fake_sim)), labelv)
                errD_fake.backward(retain_graph=True)  # backward on score on real

                labelv = Variable(label.fill_(real_label))
                errD_real = criterion_GAN(netD(Variable(real_sim)), labelv)
                errD_real.backward(retain_graph=True)  # backward on score on real

                errD = errD_real + errD_fake

                # gradient penalty
                alpha = torch.rand(opt.batchSize, 1, 1, 1).cuda()
                x_hat = Variable(alpha * real_sim + (1 - alpha) * (
                    real_sim + 0.5 * real_sim.std() * torch.rand(real_sim.size()).cuda()), requires_grad=True)
                pred_hat = F.sigmoid(netD(x_hat))
                # pred_hat = netD(x_hat)
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
                gradient_penalty = lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                gradient_penalty.backward()
            optimizerD.step()
        ############################
        # (2) Update G network
        ############################
        if iter_count < len(dataloader_train):

            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netG.parameters():
                p.requires_grad = True  # to avoid computation
            netG.zero_grad()

            data = data_iter.next()
            iter_count += 1

            if opt.cuda:
                real_sim = data.cuda()

            if flag:  # fix samples
                viz.images(
                    real_sim.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='sharp img', caption='level ')
                )
                vutils.save_image(real_sim.mul(0.5).add(0.5),
                                  '%s/sharp_samples' % opt.outf + '.png')
                flag -= 1

            if opt.gp:
                noise.normal_(0, 1)
                fake = netG(Variable(noise))
                errG = netD(fake).mean(0).view(1)
                errG.backward(mone)
            else:
                labelv = Variable(label.fill_(real_label))
                noise.normal_(0, 1)
                fake = netG(Variable(noise))
                errG = criterion_GAN(netD(fake), labelv)
                errG.backward()

            optimizerG.step()

        ############################
        # (3) Report & 100 Batch checkpoint
        ############################

        if flag4:
            D1 = viz.line(
                np.array([errD.data[0]]), np.array([gen_iterations]),
                opts=dict(title='errD(distinguishability)', caption='total Dloss')
            )
            D2 = viz.line(
                np.array([errD_real.data[0]]), np.array([gen_iterations]),
                opts=dict(title='errD_real', caption='real\'s mistake')
            )
            D3 = viz.line(
                np.array([errD_fake.data[0]]), np.array([gen_iterations]),
                opts=dict(title='errD_fake', caption='fake\'s mistake')
            )
            G1 = viz.line(
                np.array([errG.data[0]]), np.array([gen_iterations]),
                opts=dict(title='Gnet loss toward real', caption='Gnet loss')
            )
            G2 = viz.line(
                np.array([gradient_penalty.data[0]]), np.array([gen_iterations]),
                opts=dict(title='gradient_penalty', caption='gradient_penalty')
            )
            flag4 -= 1

        viz.line(np.array([errD.data[0]]), np.array([gen_iterations]), update='append', win=D1)
        viz.line(np.array([errD_real.data[0]]), np.array([gen_iterations]), update='append', win=D2)
        viz.line(np.array([errD_fake.data[0]]), np.array([gen_iterations]), update='append', win=D3)
        viz.line(np.array([errG.data[0]]), np.array([gen_iterations]), update='append', win=G1)
        viz.line(np.array([gradient_penalty.data[0]]), np.array([gen_iterations]), update='append', win=G2)

        print('[%d/%d][%d/%d][%d] errD: %f err_G: %f err_D_real: %f err_D_fake %f gp %f'
              % (epoch, opt.niter, iter_count, len(dataloader_train), gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0], gradient_penalty.data[0]))

        if gen_iterations % 100 == 0:
            fake = netG(Variable(fixed_noise, volatile=True))

            if flag3:
                imageW = viz.images(
                    fake.data.mul(0.5).add(0.5).cpu().numpy(),
                    opts=dict(title='gen img', caption='level ')
                )

                flag3 -= 1
            else:
                viz.images(
                    fake.data.mul(0.5).add(0.5).cpu().numpy(),
                    win=imageW,
                    opts=dict(title='gen img', caption='level ')
                )

        if gen_iterations % 1000 == 0:
            vutils.save_image(fake.data.mul(0.5).add(0.5),
                              '%s/fake_samples_gen_iter_%08d.png' % (opt.outf, gen_iterations))

        gen_iterations += 1

    # do checkpointing
    if opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_only.pth' % opt.outf)
        torch.save(netD.state_dict(), '%s/netD_epoch_only.pth' % opt.outf)
    elif epoch % opt.cut == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
