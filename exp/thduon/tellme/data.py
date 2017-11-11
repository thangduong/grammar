import os
import numpy as np
class TellmeData:
    def __init__(self, tellme_datadir='/mnt/work/tellme/data', datafiles = ['trn1.npy', 'trn2.npy']):
        self._tcid_count = 2
        ticid_mapfile = os.path.join(tellme_datadir, "tcid.map")
        with open(ticid_mapfile, "r") as f:
            for line in f:
                value = line.strip("\r\n").split("\t")
                if (int(value[1]) > 1):
                    self._tcid_count += 1
            self._tcid_count += 1
        self._data_chunks = []
        print("self._tcid_count = %s" % self._tcid_count)
        print("Loading data...")
        self._tcids_data = np.load(os.path.join(tellme_datadir, datafiles[0]))
        self._timing_info_data = np.load(os.path.join(tellme_datadir, datafiles[1]))
        print("done loading data!")
        self._current_epoch = 0
        self._current_index = 0
        self._num_minibatches = 0

    def current_epoch(self):
        return self._current_epoch

    def current_index(self):
        return self._current_index

    def get_tcid_count(self):
        return self._tcid_count

    def next_batch(self, batch_size=2, params=None):
        tcids_before = np.empty((0,20))
        tcid_after = np.empty((0))
        timing_info = np.empty((0,20))
        while (len(tcid_after) < batch_size):
            end_idx = self._current_index + batch_size
            if end_idx > len(self._tcids_data):
                end_idx = len(self._tcids_data)
                next_current_idx = 0
                self._current_epoch += 1
            else:
                next_current_idx = end_idx
            timing_info = np.concatenate((timing_info, self._timing_info_data[self._current_index:end_idx, :]))
            tcids_before = np.concatenate((tcids_before, self._tcids_data[self._current_index:end_idx, 0:20]))
            tcid_after = np.concatenate((tcid_after, self._tcids_data[self._current_index:end_idx, 20]))
            self._current_index = next_current_idx
        self._num_minibatches += 1
        return {'tcids_before': tcids_before, 'timing_info': timing_info, 'y':tcid_after}


if __name__ == "__main__":
    td = TellmeData()
    print(td.next_batch())
