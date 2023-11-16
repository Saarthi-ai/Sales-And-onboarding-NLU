# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from saarthi_train.postprocessing.inverse_text_normalization.hi.graph_utils import GraphFst


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")
        # protocol, username, password, domain,port, path, query_string, fragment_id     protocol://username:password@domain:port/path?query_string#fragment_id
