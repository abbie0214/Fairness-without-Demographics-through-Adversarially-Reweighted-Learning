from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser()



    parser.add_argument('--lr_primary', type=float, default=0.001,
                                            help = 'learning rate for encoder')


    parser.add_argument('--lr_adversary', type=float, default=0.1,
                                            help = 'learning rate for gaussian parameter net')

    parser.add_argument('--num_epochs', type=int, default= 1000 ,
                                            help = 'number of epochs')

    parser.add_argument('--indim', type=int, default=12,
                                            help = 'input dimension')

    parser.add_argument('--batch_size', type=int, default=512,
                                            help = 'input dimension')

    args = parser.parse_args()

    return args