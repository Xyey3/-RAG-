# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "1.0.4rc1"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')
import os
import logging

from ncastle.common import GraphDAG
from ncastle.metrics import MetricsDAG


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def _import_algo(algo):
    """
    import algorithm corresponding to `algo`

    Parameters
    ----------
    algo: str
        lowercase letters of the algorithm `algo`

    Returns
    -------
    out: class object
        castle algorithm
    """

    if algo.lower() == 'pc':
        from ncastle.algorithms import PC as Algorithm
    elif algo.lower() == 'anm':
        from ncastle.algorithms import ANMNonlinear as Algorithm
    elif algo.lower() == 'icalingam':
        from ncastle.algorithms import ICALiNGAM as Algorithm
    elif algo.lower() == 'directlingam':
        from ncastle.algorithms import DirectLiNGAM as Algorithm
    elif algo.lower() == 'notears':
        from ncastle.algorithms import Notears as Algorithm
    elif algo.lower() == 'notearslowrank':
        from ncastle.algorithms import NotearsLowRank as Algorithm
    elif algo.lower() == 'notearsnonlinear':
        from ncastle.algorithms import NotearsNonlinear as Algorithm
    elif algo.lower() == 'corl':
        from ncastle.algorithms import CORL as Algorithm
    elif algo.lower() == 'rl':
        from ncastle.algorithms import RL as Algorithm
    elif algo.lower() == 'gae':
        from ncastle.algorithms import GAE as Algorithm
    elif algo.lower() == 'ges':
        from ncastle.algorithms import GES as Algorithm
    elif algo.lower() == 'golem':
        from ncastle.algorithms import GOLEM as Algorithm
    elif algo.lower() == 'grandag':
        from ncastle.algorithms import GraNDAG as Algorithm
    elif algo.lower() == 'pnl':
        from ncastle.algorithms import PNL as Algorithm
    else:
        raise ValueError('Unknown algorithm.==========')

    logging.info(f"import algorithm corresponding to {algo} complete!")

    return Algorithm
