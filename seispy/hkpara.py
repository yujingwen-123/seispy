from os.path import expanduser
import configparser
import numpy as np
from seispy.utils import check_path, array_instance

_BOOL_STATES = {
    '1': True, 'yes': True, 'true': True, 'on': True, 'y': True, 't': True,
    '0': False, 'no': False, 'false': False, 'off': False, 'n': False, 'f': False
}


class HKPara(object):
    def __init__(self):
        self.rfpath = expanduser('~')
        self.hkpath = expanduser('~')
        self.hklist = 'hk.dat'
        self.hrange = np.arange(20, 80, 0.1)
        self.krange = np.arange(1.6, 1.9, 0.01)
        self.vp = 6.3
        self.weight = (0.7, 0.2, 0.1)
        self.plot_final_only = False
        self.energy_grid = ''
    
    def __str__(self):
        head = ['{}: {}'.format(k, v) for k, v in self.__dict__.items()]
        return '\n'.join(head)

    @property
    def hrange(self):
        return self._hrange

    @hrange.setter
    def hrange(self, value):
        if not (array_instance(value) or value is None):
            raise TypeError('Error type of hrange')
        else:
            self._hrange = value

    @property
    def krange(self):
        return self._krange

    @krange.setter
    def krange(self, value):
        if not (array_instance(value) or value is None):
            raise TypeError('Error type of krange')
        else:
            self._krange = value


def hkpara(cfg_file):
    hpara = HKPara()
    cf = configparser.ConfigParser()
    try:
        cf.read(cfg_file)
    except Exception:
        raise FileNotFoundError('Cannot open configure file %s' % cfg_file)

    # para for FileIO section
    hpara.rfpath = check_path('rfpath', cf.get('FileIO', 'rfpath'))
    hpara.hkpath = cf.get('FileIO', 'hkpath')
    hpara.hklist = cf.get('FileIO', 'hklst')

    hmin = cf.getfloat('hk', 'hmin')
    hmax = cf.getfloat('hk', 'hmax')
    kmin = cf.getfloat('hk', 'kmin')
    kmax = cf.getfloat('hk', 'kmax')
    hpara.hrange = np.arange(hmin, hmax, 0.1)
    hpara.krange = np.arange(kmin, kmax, 0.01)

    vp = cf.get('hk', 'vp')
    if vp != '':
        hpara.vp = float(vp)

    w1 = cf.getfloat('hk', 'weight1')
    w2 = cf.getfloat('hk', 'weight2')
    w3 = cf.getfloat('hk', 'weight3')
    hpara.weight = (w1, w2, w3)

    plot_final_only = cf.get('hk', 'plot_final_only', fallback='false').strip().lower()
    if plot_final_only in _BOOL_STATES:
        hpara.plot_final_only = _BOOL_STATES[plot_final_only]
    else:
        raise ValueError('Invalid boolean value for plot_final_only: {}'.format(plot_final_only))
    hpara.energy_grid = cf.get('hk', 'energy_grid', fallback='')
    return hpara
