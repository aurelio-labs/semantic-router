Local Dynamic Routes
====================

Fully local Semantic Router with ``llama.cpp`` and HuggingFace Encoder
----------------------------------------------------------------------

There are many reasons users might choose to roll their own LLMs rather
than use a third-party service. Whether it's due to cost, privacy or
compliance, Semantic Router supports the use of “local” LLMs through
``llama.cpp``.

Using ``llama.cpp`` also enables the use of quantized GGUF models,
reducing the memory footprint of deployed models, allowing even
13-billion parameter models to run with hardware acceleration on an
Apple M1 Pro chip.

Full Example
------------

|Open In Colab| |Open nbviewer|

.. |Open In Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb
.. |Open nbviewer| image:: https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg
   :target: https://nbviewer.org/github/aurelio-labs/semantic-router/blob/main/docs/05-local-execution.ipynb

Below is an example of using semantic router with
**Mistral-7B-Instruct**, quantized i.

Installing the library
----------------------

   Note: if you require hardware acceleration via BLAS, CUDA, Metal,
   etc. please refer to the
   `abetlen/llama-cpp-python <https://github.com/abetlen/llama-cpp-python#installation-with-specific-hardware-acceleration-blas-cuda-metal-etc>`__
   repository README.md

.. code:: ipython3

    !pip install -qU "semantic-router[local]"

If you’re running on Apple silicon you can run the following to run with
Metal hardware acceleration:

.. code:: ipython3

    !CMAKE_ARGS="-DLLAMA_METAL=on"


.. parsed-literal::

    'CMAKE_ARGS' is not recognized as an internal or external command,
    operable program or batch file.


Download the Mistral 7B Instruct 4-bit GGUF files
-------------------------------------------------

We will be using Mistral 7B Instruct, quantized as a 4-bit GGUF file, a
good balance between performance and ability to deploy on consumer
hardware

.. code:: ipython3

    ! curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_0.gguf?download=true" -o ./mistral-7b-instruct-v0.2.Q4_0.gguf
    ! ls mistral-7b-instruct-v0.2.Q4_0.gguf


.. parsed-literal::

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100  1168  100  1168    0     0   4171      0 --:--:-- --:--:-- --:--:--  4186
    
      0 3918M    0 6382k    0     0  7990k      0  0:08:22 --:--:--  0:08:22 7990k
      0 3918M    0 35.0M    0     0  19.4M      0  0:03:21  0:00:01  0:03:20 28.7M
      2 3918M    2 78.7M    0     0  28.1M      0  0:02:19  0:00:02  0:02:17 36.2M
      3 3918M    3  123M    0     0  32.4M      0  0:02:00  0:00:03  0:01:57 39.0M
      4 3918M    4  171M    0     0  35.6M      0  0:01:49  0:00:04  0:01:45 41.2M
      5 3918M    5  219M    0     0  37.7M      0  0:01:43  0:00:05  0:01:38 42.5M
      6 3918M    6  271M    0     0  39.9M      0  0:01:38  0:00:06  0:01:32 47.3M
      8 3918M    8  326M    0     0  41.8M      0  0:01:33  0:00:07  0:01:26 49.4M
      9 3918M    9  379M    0     0  42.5M      0  0:01:31  0:00:08  0:01:23 50.1M
     10 3918M   10  428M    0     0  43.7M      0  0:01:29  0:00:09  0:01:20 51.5M
     12 3918M   12  490M    0     0  45.3M      0  0:01:26  0:00:10  0:01:16 54.2M
     13 3918M   13  543M    0     0  46.0M      0  0:01:25  0:00:11  0:01:14 54.4M
     15 3918M   15  609M    0     0  47.6M      0  0:01:22  0:00:12  0:01:10 56.6M
     17 3918M   17  672M    0     0  48.7M      0  0:01:20  0:00:13  0:01:07 59.8M
     18 3918M   18  733M    0     0  49.5M      0  0:01:19  0:00:14  0:01:05 60.8M
     20 3918M   20  800M    0     0  50.6M      0  0:01:17  0:00:15  0:01:02 62.0M
     21 3918M   21  861M    0     0  51.2M      0  0:01:16  0:00:16  0:01:00 63.4M
     23 3918M   23  923M    0     0  51.8M      0  0:01:15  0:00:17  0:00:58 62.7M
     24 3918M   24  973M    0     0  51.7M      0  0:01:15  0:00:18  0:00:57 60.1M
     26 3918M   26 1039M    0     0  52.4M      0  0:01:14  0:00:19  0:00:55 61.2M
     28 3918M   28 1101M    0     0  52.9M      0  0:01:14  0:00:20  0:00:54 60.1M
     29 3918M   29 1156M    0     0  53.0M      0  0:01:13  0:00:21  0:00:52 59.0M
     30 3918M   30 1208M    0     0  53.0M      0  0:01:13  0:00:22  0:00:51 57.0M
     32 3918M   32 1272M    0     0  53.4M      0  0:01:13  0:00:23  0:00:50 59.9M
     34 3918M   34 1337M    0     0  53.9M      0  0:01:12  0:00:24  0:00:48 59.5M
     35 3918M   35 1397M    0     0  54.1M      0  0:01:12  0:00:25  0:00:47 59.1M
     36 3918M   36 1444M    0     0  53.9M      0  0:01:12  0:00:26  0:00:46 57.6M
     38 3918M   38 1506M    0     0  54.1M      0  0:01:12  0:00:27  0:00:45 59.2M
     40 3918M   40 1569M    0     0  54.4M      0  0:01:11  0:00:28  0:00:43 59.3M
     41 3918M   41 1630M    0     0  54.7M      0  0:01:11  0:00:29  0:00:42 58.7M
     43 3918M   43 1697M    0     0  55.1M      0  0:01:11  0:00:30  0:00:41 60.0M
     44 3918M   44 1759M    0     0  55.3M      0  0:01:10  0:00:31  0:00:39 63.0M
     45 3918M   45 1798M    0     0  54.8M      0  0:01:11  0:00:32  0:00:39 58.6M
     47 3918M   47 1866M    0     0  55.2M      0  0:01:10  0:00:33  0:00:37 59.4M
     49 3918M   49 1930M    0     0  55.4M      0  0:01:10  0:00:34  0:00:36 59.9M
     50 3918M   50 1994M    0     0  55.7M      0  0:01:10  0:00:35  0:00:35 59.4M
     52 3918M   52 2062M    0     0  56.0M      0  0:01:09  0:00:36  0:00:33 60.4M
     54 3918M   54 2121M    0     0  56.1M      0  0:01:09  0:00:37  0:00:32 64.6M
     55 3918M   55 2186M    0     0  56.3M      0  0:01:09  0:00:38  0:00:31 63.9M
     57 3918M   57 2248M    0     0  56.5M      0  0:01:09  0:00:39  0:00:30 63.6M
     59 3918M   59 2317M    0     0  56.8M      0  0:01:08  0:00:40  0:00:28 64.6M
     60 3918M   60 2382M    0     0  57.0M      0  0:01:08  0:00:41  0:00:27 64.1M
     62 3918M   62 2435M    0     0  56.8M      0  0:01:08  0:00:42  0:00:26 62.7M
     63 3918M   63 2496M    0     0  56.9M      0  0:01:08  0:00:43  0:00:25 62.0M
     64 3918M   64 2546M    0     0  56.8M      0  0:01:08  0:00:44  0:00:24 59.5M
     66 3918M   66 2596M    0     0  56.6M      0  0:01:09  0:00:45  0:00:24 55.7M
     67 3918M   67 2646M    0     0  56.5M      0  0:01:09  0:00:46  0:00:23 52.7M
     69 3918M   69 2708M    0     0  56.6M      0  0:01:09  0:00:47  0:00:22 54.7M
     70 3918M   70 2773M    0     0  56.8M      0  0:01:08  0:00:48  0:00:20 55.4M
     72 3918M   72 2833M    0     0  56.9M      0  0:01:08  0:00:49  0:00:19 57.4M
     73 3918M   73 2896M    0     0  56.9M      0  0:01:08  0:00:50  0:00:18 59.7M
     75 3918M   75 2952M    0     0  56.9M      0  0:01:08  0:00:51  0:00:17 61.1M
     76 3918M   76 3012M    0     0  57.0M      0  0:01:08  0:00:52  0:00:16 60.3M
     78 3918M   78 3078M    0     0  57.2M      0  0:01:08  0:00:53  0:00:15 60.9M
     80 3918M   80 3140M    0     0  57.3M      0  0:01:08  0:00:54  0:00:14 61.3M
     81 3918M   81 3196M    0     0  57.2M      0  0:01:08  0:00:55  0:00:13 60.1M
     83 3918M   83 3261M    0     0  57.4M      0  0:01:08  0:00:56  0:00:12 61.7M
     84 3918M   84 3324M    0     0  57.5M      0  0:01:08  0:00:57  0:00:11 62.8M
     86 3918M   86 3390M    0     0  57.6M      0  0:01:07  0:00:58  0:00:09 62.4M
     88 3918M   88 3456M    0     0  57.7M      0  0:01:07  0:00:59  0:00:08 63.1M
     89 3918M   89 3520M    0     0  57.9M      0  0:01:07  0:01:00  0:00:07 64.8M
     91 3918M   91 3585M    0     0  58.0M      0  0:01:07  0:01:01  0:00:06 64.9M
     93 3918M   93 3654M    0     0  58.1M      0  0:01:07  0:01:02  0:00:05 65.8M
     94 3918M   94 3719M    0     0  58.2M      0  0:01:07  0:01:03  0:00:04 65.7M
     96 3918M   96 3778M    0     0  58.3M      0  0:01:07  0:01:04  0:00:03 64.5M
     97 3918M   97 3828M    0     0  58.1M      0  0:01:07  0:01:05  0:00:02 61.6M
     99 3918M   99 3888M    0     0  58.2M      0  0:01:07  0:01:06  0:00:01 60.6M
    100 3918M  100 3918M    0     0  58.2M      0  0:01:07  0:01:07 --:--:-- 58.8M
    'ls' is not recognized as an internal or external command,
    operable program or batch file.


Initializing Dynamic Routes
---------------------------

Similar to the ``02-dynamic-routes.ipynb`` notebook, we will be
initializing some dynamic routes that make use of LLMs for function
calling

.. code:: ipython3

    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    from semantic_router import Route
    from semantic_router.utils.function_call import get_schema
    
    
    def get_time(timezone: str) -> str:
        """Finds the current time in a specific timezone.
    
        :param timezone: The timezone to find the current time in, should
            be a valid timezone from the IANA Time Zone Database like
            "America/New_York" or "Europe/London". Do NOT put the place
            name itself like "rome", or "new york", you must provide
            the IANA format.
        :type timezone: str
        :return: The current time in the specified timezone."""
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime("%H:%M")
    
    
    time_schema = get_schema(get_time)
    time_schema
    time = Route(
        name="get_time",
        utterances=[
            "what is the time in new york city?",
            "what is the time in london?",
            "I live in Rome, what time is it?",
        ],
        function_schemas=[time_schema],
    )
    
    politics = Route(
        name="politics",
        utterances=[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president" "don't you just hate the president",
            "they're going to destroy this country!",
            "they will save the country!",
        ],
    )
    chitchat = Route(
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "how are things going?",
            "lovely weather today",
            "the weather is horrendous",
            "let's go to the chippy",
        ],
    )
    
    routes = [politics, chitchat, time]


.. parsed-literal::

    c:\Users\Siraj\Documents\Personal\Work\Aurelio\Virtual Environments\semantic_router_3\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


.. code:: ipython3

    time_schema




.. parsed-literal::

    {'name': 'get_time',
     'description': 'Finds the current time in a specific timezone.\n\n:param timezone: The timezone to find the current time in, should\n    be a valid timezone from the IANA Time Zone Database like\n    "America/New_York" or "Europe/London". Do NOT put the place\n    name itself like "rome", or "new york", you must provide\n    the IANA format.\n:type timezone: str\n:return: The current time in the specified timezone.',
     'signature': '(timezone: str) -> str',
     'output': "<class 'str'>"}



Encoders
--------

You can use alternative Encoders, however, in this example we want to
showcase a fully-local Semantic Router execution, so we are going to use
a ``HuggingFaceEncoder`` with ``sentence-transformers/all-MiniLM-L6-v2``
(the default) as an embedding model.

.. code:: ipython3

    from semantic_router.encoders import HuggingFaceEncoder
    
    encoder = HuggingFaceEncoder()

``llama.cpp`` LLM
-----------------

From here, we can go ahead and instantiate our ``llama-cpp-python``
``llama_cpp.Llama`` LLM, and then pass it to the
``semantic_router.llms.LlamaCppLLM`` wrapper class.

For ``llama_cpp.Llama``, there are a couple of parameters you should pay
attention to:

-  ``n_gpu_layers``: how many LLM layers to offload to the GPU (if you
   want to offload the entire model, pass ``-1``, and for CPU execution,
   pass ``0``)
-  ``n_ctx``: context size, limit the number of tokens that can be
   passed to the LLM (this is bounded by the model’s internal maximum
   context size, in this case for Mistral-7B-Instruct, 8000 tokens)
-  ``verbose``: if ``False``, silences output from ``llama.cpp``

..

   For other parameter explanation, refer to the ``llama-cpp-python``
   `API
   Reference <https://llama-cpp-python.readthedocs.io/en/latest/api-reference/>`__

.. code:: ipython3

    from semantic_router import RouteLayer
    
    from llama_cpp import Llama
    from semantic_router.llms.llamacpp import LlamaCppLLM
    
    enable_gpu = True  # offload LLM layers to the GPU (must fit in memory)
    
    _llm = Llama(
        model_path="./mistral-7b-instruct-v0.2.Q4_0.gguf",
        n_gpu_layers=-1 if enable_gpu else 0,
        n_ctx=2048,
    )
    _llm.verbose = False
    llm = LlamaCppLLM(name="Mistral-7B-v0.2-Instruct", llm=_llm, max_tokens=None)
    
    rl = RouteLayer(encoder=encoder, routes=routes, llm=llm)


.. parsed-literal::

    llama_model_loader: loaded meta data with 24 key-value pairs and 291 tensors from ./mistral-7b-instruct-v0.2.Q4_0.gguf (version GGUF V3 (latest))
    llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
    llama_model_loader: - kv   0:                       general.architecture str              = llama
    llama_model_loader: - kv   1:                               general.name str              = mistralai_mistral-7b-instruct-v0.2
    llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
    llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
    llama_model_loader: - kv   4:                          llama.block_count u32              = 32
    llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
    llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
    llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
    llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
    llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
    llama_model_loader: - kv  10:                       llama.rope.freq_base f32              = 1000000.000000
    llama_model_loader: - kv  11:                          general.file_type u32              = 2
    llama_model_loader: - kv  12:                       tokenizer.ggml.model str              = llama
    llama_model_loader: - kv  13:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
    llama_model_loader: - kv  14:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
    llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
    llama_model_loader: - kv  16:                tokenizer.ggml.bos_token_id u32              = 1
    llama_model_loader: - kv  17:                tokenizer.ggml.eos_token_id u32              = 2
    llama_model_loader: - kv  18:            tokenizer.ggml.unknown_token_id u32              = 0
    llama_model_loader: - kv  19:            tokenizer.ggml.padding_token_id u32              = 0
    llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = true
    llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
    llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
    llama_model_loader: - kv  23:               general.quantization_version u32              = 2
    llama_model_loader: - type  f32:   65 tensors
    llama_model_loader: - type q4_0:  225 tensors
    llama_model_loader: - type q6_K:    1 tensors
    llm_load_vocab: special tokens definition check successful ( 259/32000 ).
    llm_load_print_meta: format           = GGUF V3 (latest)
    llm_load_print_meta: arch             = llama
    llm_load_print_meta: vocab type       = SPM
    llm_load_print_meta: n_vocab          = 32000
    llm_load_print_meta: n_merges         = 0
    llm_load_print_meta: n_ctx_train      = 32768
    llm_load_print_meta: n_embd           = 4096
    llm_load_print_meta: n_head           = 32
    llm_load_print_meta: n_head_kv        = 8
    llm_load_print_meta: n_layer          = 32
    llm_load_print_meta: n_rot            = 128
    llm_load_print_meta: n_embd_head_k    = 128
    llm_load_print_meta: n_embd_head_v    = 128
    llm_load_print_meta: n_gqa            = 4
    llm_load_print_meta: n_embd_k_gqa     = 1024
    llm_load_print_meta: n_embd_v_gqa     = 1024
    llm_load_print_meta: f_norm_eps       = 0.0e+00
    llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
    llm_load_print_meta: f_clamp_kqv      = 0.0e+00
    llm_load_print_meta: f_max_alibi_bias = 0.0e+00
    llm_load_print_meta: f_logit_scale    = 0.0e+00
    llm_load_print_meta: n_ff             = 14336
    llm_load_print_meta: n_expert         = 0
    llm_load_print_meta: n_expert_used    = 0
    llm_load_print_meta: causal attn      = 1
    llm_load_print_meta: pooling type     = 0
    llm_load_print_meta: rope type        = 0
    llm_load_print_meta: rope scaling     = linear
    llm_load_print_meta: freq_base_train  = 1000000.0
    llm_load_print_meta: freq_scale_train = 1
    llm_load_print_meta: n_yarn_orig_ctx  = 32768
    llm_load_print_meta: rope_finetuned   = unknown
    llm_load_print_meta: ssm_d_conv       = 0
    llm_load_print_meta: ssm_d_inner      = 0
    llm_load_print_meta: ssm_d_state      = 0
    llm_load_print_meta: ssm_dt_rank      = 0
    llm_load_print_meta: model type       = 8B
    llm_load_print_meta: model ftype      = Q4_0
    llm_load_print_meta: model params     = 7.24 B
    llm_load_print_meta: model size       = 3.83 GiB (4.54 BPW) 
    llm_load_print_meta: general.name     = mistralai_mistral-7b-instruct-v0.2
    llm_load_print_meta: BOS token        = 1 '<s>'
    llm_load_print_meta: EOS token        = 2 '</s>'
    llm_load_print_meta: UNK token        = 0 '<unk>'
    llm_load_print_meta: PAD token        = 0 '<unk>'
    llm_load_print_meta: LF token         = 13 '<0x0A>'
    llm_load_tensors: ggml ctx size =    0.15 MiB
    llm_load_tensors:        CPU buffer size =  3917.87 MiB
    ..................................................................................................
    llama_new_context_with_model: n_ctx      = 2048
    llama_new_context_with_model: n_batch    = 512
    llama_new_context_with_model: n_ubatch   = 512
    llama_new_context_with_model: flash_attn = 0
    llama_new_context_with_model: freq_base  = 1000000.0
    llama_new_context_with_model: freq_scale = 1
    llama_kv_cache_init:        CPU KV buffer size =   256.00 MiB
    llama_new_context_with_model: KV self size  =  256.00 MiB, K (f16):  128.00 MiB, V (f16):  128.00 MiB
    llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
    llama_new_context_with_model:        CPU compute buffer size =   164.01 MiB
    llama_new_context_with_model: graph nodes  = 1030
    llama_new_context_with_model: graph splits = 1
    AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
    Model metadata: {'general.name': 'mistralai_mistral-7b-instruct-v0.2', 'general.architecture': 'llama', 'llama.context_length': '32768', 'llama.rope.dimension_count': '128', 'llama.embedding_length': '4096', 'llama.block_count': '32', 'llama.feed_forward_length': '14336', 'llama.attention.head_count': '32', 'tokenizer.ggml.eos_token_id': '2', 'general.file_type': '2', 'llama.attention.head_count_kv': '8', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.rope.freq_base': '1000000.000000', 'tokenizer.ggml.model': 'llama', 'general.quantization_version': '2', 'tokenizer.ggml.bos_token_id': '1', 'tokenizer.ggml.unknown_token_id': '0', 'tokenizer.ggml.padding_token_id': '0', 'tokenizer.ggml.add_bos_token': 'true', 'tokenizer.ggml.add_eos_token': 'false', 'tokenizer.chat_template': "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"}
    Guessed chat format: mistral-instruct
    [32m2024-05-07 15:50:07 INFO semantic_router.utils.logger local[0m


.. code:: ipython3

    rl("how's the weather today?")




.. parsed-literal::

    RouteChoice(name='chitchat', function_call=None, similarity_score=None)



.. code:: ipython3

    out = rl("what's the time in New York right now?")
    print(out)
    get_time(**out.function_call[0])


.. parsed-literal::

    from_string grammar:
    root ::= object 
    object ::= [{] ws object_11 [}] ws 
    value ::= object | array | string | number | value_6 ws 
    array ::= [[] ws array_15 []] ws 
    string ::= ["] string_18 ["] ws 
    number ::= number_19 number_25 number_29 ws 
    value_6 ::= [t] [r] [u] [e] | [f] [a] [l] [s] [e] | [n] [u] [l] [l] 
    ws ::= ws_31 
    object_8 ::= string [:] ws value object_10 
    object_9 ::= [,] ws string [:] ws value 
    object_10 ::= object_9 object_10 | 
    object_11 ::= object_8 | 
    array_12 ::= value array_14 
    array_13 ::= [,] ws value 
    array_14 ::= array_13 array_14 | 
    array_15 ::= array_12 | 
    string_16 ::= [^"\] | [\] string_17 
    string_17 ::= ["\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] 
    string_18 ::= string_16 string_18 | 
    number_19 ::= number_20 number_21 
    number_20 ::= [-] | 
    number_21 ::= [0-9] | [1-9] number_22 
    number_22 ::= [0-9] number_22 | 
    number_23 ::= [.] number_24 
    number_24 ::= [0-9] number_24 | [0-9] 
    number_25 ::= number_23 | 
    number_26 ::= [eE] number_27 number_28 
    number_27 ::= [-+] | 
    number_28 ::= [0-9] number_28 | [0-9] 
    number_29 ::= number_26 | 
    ws_30 ::= [ <U+0009><U+000A>] ws 
    ws_31 ::= ws_30 | 
    
    [32m2024-05-07 15:50:08 INFO semantic_router.utils.logger Extracting function input...[0m
    [32m2024-05-07 15:50:59 INFO semantic_router.utils.logger LLM output: {
    	"timezone": "America/New_York"
    }[0m
    [32m2024-05-07 15:50:59 INFO semantic_router.utils.logger Function inputs: [{'timezone': 'America/New_York'}][0m


.. parsed-literal::

    name='get_time' function_call=[{'timezone': 'America/New_York'}] similarity_score=None




.. parsed-literal::

    '07:50'



.. code:: ipython3

    out = rl("what's the time in Rome right now?")
    print(out)
    get_time(**out.function_call[0])


.. parsed-literal::

    from_string grammar:
    root ::= object 
    object ::= [{] ws object_11 [}] ws 
    value ::= object | array | string | number | value_6 ws 
    array ::= [[] ws array_15 []] ws 
    string ::= ["] string_18 ["] ws 
    number ::= number_19 number_25 number_29 ws 
    value_6 ::= [t] [r] [u] [e] | [f] [a] [l] [s] [e] | [n] [u] [l] [l] 
    ws ::= ws_31 
    object_8 ::= string [:] ws value object_10 
    object_9 ::= [,] ws string [:] ws value 
    object_10 ::= object_9 object_10 | 
    object_11 ::= object_8 | 
    array_12 ::= value array_14 
    array_13 ::= [,] ws value 
    array_14 ::= array_13 array_14 | 
    array_15 ::= array_12 | 
    string_16 ::= [^"\] | [\] string_17 
    string_17 ::= ["\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] 
    string_18 ::= string_16 string_18 | 
    number_19 ::= number_20 number_21 
    number_20 ::= [-] | 
    number_21 ::= [0-9] | [1-9] number_22 
    number_22 ::= [0-9] number_22 | 
    number_23 ::= [.] number_24 
    number_24 ::= [0-9] number_24 | [0-9] 
    number_25 ::= number_23 | 
    number_26 ::= [eE] number_27 number_28 
    number_27 ::= [-+] | 
    number_28 ::= [0-9] number_28 | [0-9] 
    number_29 ::= number_26 | 
    ws_30 ::= [ <U+0009><U+000A>] ws 
    ws_31 ::= ws_30 | 
    
    [32m2024-05-07 15:50:59 INFO semantic_router.utils.logger Extracting function input...[0m
    [32m2024-05-07 15:51:27 INFO semantic_router.utils.logger LLM output: {"timezone": "Europe/Rome"}[0m
    [32m2024-05-07 15:51:27 INFO semantic_router.utils.logger Function inputs: [{'timezone': 'Europe/Rome'}][0m


.. parsed-literal::

    name='get_time' function_call=[{'timezone': 'Europe/Rome'}] similarity_score=None




.. parsed-literal::

    '13:51'



.. code:: ipython3

    out = rl("what's the time in Bangkok right now?")
    print(out)
    get_time(**out.function_call[0])


.. parsed-literal::

    from_string grammar:
    root ::= object 
    object ::= [{] ws object_11 [}] ws 
    value ::= object | array | string | number | value_6 ws 
    array ::= [[] ws array_15 []] ws 
    string ::= ["] string_18 ["] ws 
    number ::= number_19 number_25 number_29 ws 
    value_6 ::= [t] [r] [u] [e] | [f] [a] [l] [s] [e] | [n] [u] [l] [l] 
    ws ::= ws_31 
    object_8 ::= string [:] ws value object_10 
    object_9 ::= [,] ws string [:] ws value 
    object_10 ::= object_9 object_10 | 
    object_11 ::= object_8 | 
    array_12 ::= value array_14 
    array_13 ::= [,] ws value 
    array_14 ::= array_13 array_14 | 
    array_15 ::= array_12 | 
    string_16 ::= [^"\] | [\] string_17 
    string_17 ::= ["\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] 
    string_18 ::= string_16 string_18 | 
    number_19 ::= number_20 number_21 
    number_20 ::= [-] | 
    number_21 ::= [0-9] | [1-9] number_22 
    number_22 ::= [0-9] number_22 | 
    number_23 ::= [.] number_24 
    number_24 ::= [0-9] number_24 | [0-9] 
    number_25 ::= number_23 | 
    number_26 ::= [eE] number_27 number_28 
    number_27 ::= [-+] | 
    number_28 ::= [0-9] number_28 | [0-9] 
    number_29 ::= number_26 | 
    ws_30 ::= [ <U+0009><U+000A>] ws 
    ws_31 ::= ws_30 | 
    
    [32m2024-05-07 15:51:27 INFO semantic_router.utils.logger Extracting function input...[0m
    [32m2024-05-07 15:51:56 INFO semantic_router.utils.logger LLM output: {"timezone": "Asia/Bangkok"}[0m
    [32m2024-05-07 15:51:56 INFO semantic_router.utils.logger Function inputs: [{'timezone': 'Asia/Bangkok'}][0m


.. parsed-literal::

    name='get_time' function_call=[{'timezone': 'Asia/Bangkok'}] similarity_score=None




.. parsed-literal::

    '18:51'



.. code:: ipython3

    out = rl("what's the time in Phuket right now?")
    print(out)
    get_time(**out.function_call[0])


.. parsed-literal::

    from_string grammar:
    root ::= object 
    object ::= [{] ws object_11 [}] ws 
    value ::= object | array | string | number | value_6 ws 
    array ::= [[] ws array_15 []] ws 
    string ::= ["] string_18 ["] ws 
    number ::= number_19 number_25 number_29 ws 
    value_6 ::= [t] [r] [u] [e] | [f] [a] [l] [s] [e] | [n] [u] [l] [l] 
    ws ::= ws_31 
    object_8 ::= string [:] ws value object_10 
    object_9 ::= [,] ws string [:] ws value 
    object_10 ::= object_9 object_10 | 
    object_11 ::= object_8 | 
    array_12 ::= value array_14 
    array_13 ::= [,] ws value 
    array_14 ::= array_13 array_14 | 
    array_15 ::= array_12 | 
    string_16 ::= [^"\] | [\] string_17 
    string_17 ::= ["\/bfnrt] | [u] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] 
    string_18 ::= string_16 string_18 | 
    number_19 ::= number_20 number_21 
    number_20 ::= [-] | 
    number_21 ::= [0-9] | [1-9] number_22 
    number_22 ::= [0-9] number_22 | 
    number_23 ::= [.] number_24 
    number_24 ::= [0-9] number_24 | [0-9] 
    number_25 ::= number_23 | 
    number_26 ::= [eE] number_27 number_28 
    number_27 ::= [-+] | 
    number_28 ::= [0-9] number_28 | [0-9] 
    number_29 ::= number_26 | 
    ws_30 ::= [ <U+0009><U+000A>] ws 
    ws_31 ::= ws_30 | 
    
    [32m2024-05-07 15:51:56 INFO semantic_router.utils.logger Extracting function input...[0m
    [32m2024-05-07 15:52:25 INFO semantic_router.utils.logger LLM output: {
    	"timezone": "Asia/Bangkok"
    }[0m
    [32m2024-05-07 15:52:25 INFO semantic_router.utils.logger Function inputs: [{'timezone': 'Asia/Bangkok'}][0m


.. parsed-literal::

    name='get_time' function_call=[{'timezone': 'Asia/Bangkok'}] similarity_score=None




.. parsed-literal::

    '18:52'



Cleanup
-------

Once done, if you’d like to delete the downloaded model you can do so
with the following:

::

   ! rm ./mistral-7b-instruct-v0.2.Q4_0.gguf
