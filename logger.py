import os
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy, plot_alignment
from plotting_utils import plot_gate_outputs_to_numpy
import pdb

class ParrotLogger(SummaryWriter):
    def __init__(self, logdir, ali_path='ali'):
        super(ParrotLogger, self).__init__(logdir)
        ali_path = os.path.join(logdir, ali_path)
        if not os.path.exists(ali_path):
            os.makedirs(ali_path)
        self.ali_path = ali_path

    def log_training(self, reduced_loss, reduced_losses, reduced_acces, grad_norm, learning_rate, duration,
                     iteration):
        
        self.add_scalar("training.loss", reduced_loss, iteration)

        # self.add_scalar("training.loss.spenc", reduced_losses[0], iteration)

        self.add_scalar("training.loss.texcl", reduced_losses[0], iteration)

        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

        #pdb.set_trace()
        # self.add_scalar('training.acc.spenc', reduced_acces[0], iteration)

        self.add_scalar('training.acc.texcl', reduced_acces[0], iteration)
    
    def log_validation(self, reduced_loss, reduced_losses, reduced_acces, model, y, y_pred, iteration, task):

        self.add_scalar('validation.loss.%s'%task, reduced_loss, iteration)

        # self.add_scalar("validation.loss.%s.spenc"%task, reduced_losses[0], iteration)

        self.add_scalar("validation.loss.%s.texcl"%task, reduced_losses[0], iteration)

        # self.add_scalar('validation.acc.%s.spenc'%task, reduced_acces[0], iteration)

        self.add_scalar('validation.acc.%s.texcl'%task, reduced_acces[0], iteration)
        

        # text_hidden, text_logit, \
        # speaker_logits, \
        # text_lengths, mel_lengths, SE_alignments = y_pred

        # text_target, mel_target, speaker_target,  stop_target  = y

        # idx = random.randint(0, SE_alignments.size(0) - 1)

        # SE_alignments = SE_alignments.data.cpu().numpy()


        # self.add_image(
        #     "%s.SE_alignments"%task,
        #     plot_alignment_to_numpy(SE_alignments[idx].T),
        #     iteration, dataformats='HWC')


