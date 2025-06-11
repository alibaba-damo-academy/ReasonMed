import argparse
import json
import os
import re

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm

import argparse
import json
import os
import re
import gc
import time
from tqdm import tqdm
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
import concurrent.futures
from openai import AzureOpenAI  # 导入 AzureOpenAI 库



import argparse
import json
import os
import re
import gc

from vllm import LLM, SamplingParams


import argparse
import json
import os
import gc

from vllm import LLM, SamplingParams


# from __future__ import annotations

import argparse
import json
import os
import re
import gc
from typing import Dict, List, Any

from vllm import LLM, SamplingParams


import argparse
import json
import os
import re

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm

import argparse
import json
import os
import gc
import time
from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm
import concurrent.futures  # 用于多线程并发处理


import argparse
import json
import os
import random
import re
import gc
import concurrent.futures  # 并发请求处理
import time  # 用于处理重试时的延时
from typing import List, Dict, Any
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from tqdm import tqdm  # 导入tqdm