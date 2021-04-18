import numpy as np
from torch.utils.data.sampler import RandomSampler


class Customsampler(RandomSampler):

    def __init__(self, data_source, replacement=False, num_samples=None, batch_size=None, generator=None):
        super(Customsampler, self).__init__(data_source=data_source, replacement=replacement,
                                            num_samples=num_samples, generator=generator)
        
        self.l = data_source.num_classes
        self.g = data_source.num_groups
        self.nbatch_size = batch_size // (self.l*self.g)
        self.num_data = data_source.num_data
        pos = np.unravel_index(np.argmax(self.num_data), self.num_data.shape)
        self.max_pos = pos[0] * self.g + pos[1]

    def __iter__(self):
        final_list = []
        index_list = []
        total_num = 0
        for i in range(self.l*self.g):
            tmp = np.arange(self.num_data[i//self.l, i%self.l]) + total_num
            np.random.shuffle(tmp)
            index_list.append(list(tmp))
            if i != self.max_pos:
                while len(index_list[-1]) < np.max(self.num_data):
                    tmp = np.arange(self.num_data[i//self.l, i%self.l]) + total_num
                    np.random.shuffle(tmp)                    
                    index_list[-1].extend(list(tmp))
            total_num += self.num_data[i//self.l, i%self.l]

        for tmp in range(len(index_list[self.max_pos]) // self.nbatch_size):
            for list_ in index_list:
                final_list.extend(list_[tmp*self.nbatch_size:(tmp+1)*self.nbatch_size])

        return iter(final_list)


def gen(index_list, nbatch_size):
    idx = 0
    np.random.shuffle(index_list)
    while True:
        idx += nbatch_size
        if idx > len(index_list):
            print('lets go')
            raise StopIteration
        yield index_list[idx-nbatch_size:nbatch_size]

