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
        self.hstep = 0.1
        self.kstep = 0.01
        self.vp = 6.3
        self.weight = (0.7, 0.2, 0.1)
        self.plot_final_only = False
        self.energy_grid = ''
        self.stack_method = 'linear'
        self.nth_order = 4
        self.use_bootstrap = True
        self.n_bootstrap = 100
    
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
    hpara.hstep = cf.getfloat('hk', 'hstep', fallback=0.1)
    hpara.kstep = cf.getfloat('hk', 'kstep', fallback=0.01)
    hpara.hrange = np.arange(hmin, hmax + hpara.hstep, hpara.hstep)
    hpara.krange = np.arange(kmin, kmax + hpara.kstep, hpara.kstep)

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

    # Stacking method: 'linear', 'nth_root', 'cc', 'nth_root_cc'
    stack_method = cf.get('hk', 'stack_method', fallback='linear').strip().lower()
    if stack_method in ('linear', 'nth_root', 'cc', 'nth_root_cc'):
        hpara.stack_method = stack_method
    else:
        raise ValueError('Invalid stack_method: {}. Valid options: linear, nth_root, cc, nth_root_cc'.format(stack_method))

    # Nth-root order (only used when stack_method is 'nth_root')
    nth_order = cf.get('hk', 'nth_order', fallback='4')
    hpara.nth_order = int(nth_order)

    # Bootstrap error estimation (Niu et al., 2007)
    use_bootstrap = cf.get('hk', 'use_bootstrap', fallback='true').strip().lower()
    if use_bootstrap in _BOOL_STATES:
        hpara.use_bootstrap = _BOOL_STATES[use_bootstrap]
    else:
        raise ValueError('Invalid boolean value for use_bootstrap: {}'.format(use_bootstrap))
    hpara.n_bootstrap = cf.getint('hk', 'n_bootstrap', fallback=200)

    return hpara
