// PatternRAG Benchmark Dataset (Run ID: 20260921_114121)
// Self-contained data bundle for the PatternRAG Evaluation Dashboard

window.BENCHMARK_DATA = {
  metadata: {
    run_id: "20260921_114121",
    timestamp: "2026-09-21T11:41:21",
    dataset: "HotpotQA (Multi-hop Validation Subset)",
    split: "validation",
    num_queries: 30,
    paired_runs: 60,
    llm_generator: "openai/gpt-oss-120b (via Groq API)",
    ragas_judge: "openai/gpt-oss-120b (via Groq API)",
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2 (384-dim)",
    retrieval_fusion: "Reciprocal Rank Fusion (RRF, constant k=60)",
    top_k: 10,
    completion_rate: "100% (60/60 executions completed, 0 permanent failures)"
  },

  summaryCards: [
    {
      title: "Hybrid Retrieval Equivalence",
      value: "100.0%",
      subtext: "30 / 30 paired queries",
      badge: "Identical Top-10 Ranking",
      type: "success",
      description: "Both Monolithic RRF and PatternRAG HybridRetriever yielded identical document IDs in identical rank positions."
    },
    {
      title: "Mean Reciprocal Rank (MRR)",
      value: "0.8400",
      subtext: "Monolithic: 0.8400 | PatternRAG: 0.8400",
      badge: "Δ 0.00% (Exact match)",
      type: "neutral",
      description: "Top relevant document found at identical ranking depths across the entire evaluation set."
    },
    {
      title: "NDCG @ 10",
      value: "0.8126",
      subtext: "Monolithic: 0.8126 | PatternRAG: 0.8126",
      badge: "Δ 0.00% (Exact match)",
      type: "neutral",
      description: "Normalized Discounted Cumulative Gain confirms ranking fidelity was perfectly preserved."
    },
    {
      title: "RAGAS Faithfulness",
      value: "0.4667",
      subtext: "Monolithic: 0.4500 | PatternRAG: 0.4667",
      badge: "+3.7% PatternRAG",
      type: "success",
      description: "Factual alignment of generated answers with retrieved context passages (LLM judge evaluated)."
    },
    {
      title: "RAGAS Answer Relevancy",
      value: "0.2777",
      subtext: "Monolithic: 0.2517 | PatternRAG: 0.2777",
      badge: "+10.3% PatternRAG",
      type: "success",
      description: "Semantic correspondence between query prompt and model generated answer."
    },
    {
      title: "Architectural Change Localization",
      value: "0 lines modified",
      subtext: "in core PatternRAG pipeline for M1-M3",
      badge: "Open-Closed Principle",
      type: "accent",
      description: "Extensions (TF-IDF Strategy, Filter Decorator, Complexity Observer) added without modifying core pipeline code."
    }
  ],

  retrievalMetrics: {
    aggregate: [
      { metric: "MRR", label: "Mean Reciprocal Rank", monolithic: 0.8400, patternrag: 0.8400, diff: 0.0, pctDiff: "0.0%" },
      { metric: "NDCG@10", label: "Normalized Discounted Cumulative Gain @ 10", monolithic: 0.8126, patternrag: 0.8126, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Precision@1", label: "Precision @ 1", monolithic: 0.7667, patternrag: 0.7667, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Precision@5", label: "Precision @ 5", monolithic: 0.2400, patternrag: 0.2400, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Precision@10", label: "Precision @ 10", monolithic: 0.1433, patternrag: 0.1433, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Recall@1", label: "Recall @ 1", monolithic: 0.3833, patternrag: 0.3833, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Recall@5", label: "Recall @ 5", monolithic: 0.6000, patternrag: 0.6000, diff: 0.0, pctDiff: "0.0%" },
      { metric: "Recall@10", label: "Recall @ 10", monolithic: 0.7167, patternrag: 0.7167, diff: 0.0, pctDiff: "0.0%" }
    ],
    note: "Hybrid retrieval in both architectures utilizes standard Reciprocal Rank Fusion (RRF, constant k=60) combining BM25 lexical and FAISS dense scores. Evaluated across 30 multi-hop HotpotQA queries."
  },

  ragasMetrics: {
    aggregate: [
      { metric: "faithfulness", label: "Faithfulness", monolithic: 0.4500, patternrag: 0.4667, diff: 0.0167, pctDiff: "+3.7%", interpretation: "Slight edge to PatternRAG; answers faithfully grounded in context." },
      { metric: "answer_relevancy", label: "Answer Relevancy", monolithic: 0.2517, patternrag: 0.2777, diff: 0.0260, pctDiff: "+10.3%", interpretation: "PatternRAG produced slightly higher direct relevance." },
      { metric: "context_precision", label: "Context Precision", monolithic: 0.4220, patternrag: 0.3940, diff: -0.0280, pctDiff: "-6.6%", interpretation: "Signal-to-noise ratio in retrieved context passages." },
      { metric: "context_recall", label: "Context Recall", monolithic: 0.6333, patternrag: 0.6333, diff: 0.0000, pctDiff: "0.0%", interpretation: "Identical ground-truth coverage from retrieved passages." }
    ],
    judgeModel: "openai/gpt-oss-120b (via Groq)",
    embeddingModel: "sentence-transformers/all-MiniLM-L6-v2",
    rateLimitEvents: "370 transient TPM 429 retries recovered during evaluation via exponential backoff."
  },

  latencyMetrics: {
    retrieval: {
      mean_ms: { monolithic: 966.19, patternrag: 1443.11, diff_ms: 476.92 },
      median_ms: { monolithic: 917.55, patternrag: 1333.15, diff_ms: 415.60 },
      p95_ms: { monolithic: 1739.05, patternrag: 1791.45, diff_ms: 52.40 }
    },
    generation: {
      mean_ms: { monolithic: 6497.75, patternrag: 6657.96, diff_ms: 160.21 },
      median_ms: { monolithic: 7121.17, patternrag: 7121.75, diff_ms: 0.58 },
      p95_ms: { monolithic: 10557.57, patternrag: 10566.55, diff_ms: 8.98 }
    },
    total: {
      mean_ms: { monolithic: 7464.06, patternrag: 8101.30, diff_ms: 637.24 },
      median_ms: { monolithic: 8543.66, patternrag: 8297.29, diff_ms: -246.37 },
      p95_ms: { monolithic: 11258.85, patternrag: 11800.39, diff_ms: 541.54 }
    },
    note: "External LLM generation latency accounted for >80% of total pipeline runtime. 46 transient HTTP 429 events occurred during generation with 45 automated retries successfully recovering without failure."
  },

  observerTelemetry: {
    latencyLogger: {
      retrieval_mean_ms: 1443.11,
      generation_mean_ms: 6657.96,
      queries_tracked: 30
    },
    metricsCollector: {
      total_queries: 30,
      mean_docs_retrieved: 10.0,
      top_k_configured: 10
    },
    costMonitor: {
      total_prompt_tokens: 40017,
      total_completion_tokens: 4875,
      total_tokens: 44892,
      mean_tokens_per_query: 1496.4,
      generations: 30,
      estimated_cost_usd: 0.027461
    }
  },

  architecturalExperiments: [
    {
      id: "M1",
      name: "New Retrieval Strategy (TF-IDF)",
      pattern: "Strategy Pattern",
      goal: "Add TF-IDF retrieval as an interchangeable alternative to BM25 and Dense.",
      monolithicImpact: "Required invasive edits directly inside monolithic/pipeline.py to add tfidf_index parameter, condition checks, and custom ranking branches.",
      patternRagImpact: "Created standalone class TFIDFRetriever implementing RetrieverStrategy; added one entry in PipelineFactory. Zero lines altered in patternrag/pipeline.py.",
      corePipelineModified: "0 lines (PatternRAG) vs Central Pipeline Modified (Monolithic)",
      badge: "Strategy Pattern",
      testCoverage: "test_m1_tfidf.py (+5 dedicated unit tests)"
    },
    {
      id: "M2",
      name: "Result Filtering Decorator",
      pattern: "Decorator Pattern",
      goal: "Add transparent score threshold filtering with Top-1 fallback guarantee before generation.",
      monolithicImpact: "Required intercepting retrieval outputs directly inside monolithic/pipeline.py run() loop, mixing filtering concern with generation logic.",
      patternRagImpact: "Implemented ResultFilterDecorator wrapping any RetrieverStrategy dynamically. Configurable via patternrag.yaml with zero alterations to core pipeline.",
      corePipelineModified: "0 lines (PatternRAG) vs Central Pipeline Modified (Monolithic)",
      badge: "Decorator Pattern",
      testCoverage: "test_m2_filter.py (+8 dedicated unit tests)"
    },
    {
      id: "M3",
      name: "Query Complexity Telemetry",
      pattern: "Observer Pattern",
      goal: "Track query length, token counts, and outlier complexity metrics without slowing or touching the pipeline.",
      monolithicImpact: "Required injecting tokenization, character counting, and logging code inline into the monolithic execution path.",
      patternRagImpact: "Created QueryComplexityMonitor subclassing PipelineObserver. Simply registers with EventDispatcher to listen to QUERY_RECEIVED events. Completely non-intrusive.",
      corePipelineModified: "0 lines (PatternRAG) vs Central Pipeline Modified (Monolithic)",
      badge: "Observer Pattern",
      testCoverage: "test_m3_query_complexity.py (+8 dedicated unit tests)"
    }
  ],

  queries: [
    {
      index: 1,
      id: "5ae143ed55429920d5234360",
      question: "In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?",
      gold_answer: "1755",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 987.2,
        generation_ms: 2144.3,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1346.7,
        generation_ms: 985.3,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Sergei_Aleksandrovich_Tokarev_para8", "Sergei_Sholokhov_para4", "Sergei_Aleksandrovich_Kudryavtsev_para9",
        "Sergei_Kosarev_para6", "Pavel_Alexandrov_para2", "John_Quigley_(academic)_para9",
        "Sergei_Dmitrochenko_para7", "Jan_D._Achenbach_para5", "Anatoly_Fomenko_para1", "Jeffrey_MacKie-Mason_para1"
      ]
    },
    {
      index: 2,
      id: "5abc19705542993a06baf86e",
      question: "Black Book starred the actress and writer of what heritage?",
      gold_answer: "Dutch",
      monolithic: {
        predicted_answer: "Jewish.",
        retrieval_ms: 569.0,
        generation_ms: 1128.2,
        scores: { faithfulness: 0.0, answer_relevancy: 0.128, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "",
        retrieval_ms: 1025.1,
        generation_ms: 1395.0,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "What_I_Saw:_Reports_from_Berlin,_1920-1933_para7", "Black_Book_(novel)_para3", "Black_Power:_The_Politics_of_Liberation_para7",
        "Black_Books_para3", "The_Adventures_of_the_Black_Stallion_para9", "James_Watkins_(director)_para4",
        "Black_Book_(World_War_II)_para1", "Black_Is_White_para4", "Laura_Marano_para2", "Black_13_para3"
      ]
    },
    {
      index: 3,
      id: "5ac3e0f7554299194317388b",
      question: "Which actor does American Beauty and American Beauty have in common?",
      gold_answer: "Kevin Spacey",
      monolithic: {
        predicted_answer: "",
        retrieval_ms: 377.3,
        generation_ms: 1161.3,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.35, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "",
        retrieval_ms: 990.4,
        generation_ms: 8605.8,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.35, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Houston_Beauty_para0", "American_Beauty_(1999_film)_para8", "Lush_(company)_para6",
        "American_Beauty_(1927_film)_para0", "Allure_(magazine)_para2", "La_Belle_Américaine_para3",
        "Allure_(magazine)_para5", "List_of_accolades_received_by_American_Beauty_para3", "Beauty_and_the_Beast_(1987_film)_para2", "List_of_accolades_received_by_American_Beauty_para4"
      ]
    },
    {
      index: 4,
      id: "5ae518655542993aec5ec139",
      question: "Ken Pruitt was a Republican member of an upper house of the legislature with how many members ?",
      gold_answer: "40 members",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 544.8,
        generation_ms: 2559.8,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1466.2,
        generation_ms: 6661.0,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Ken_Pruitt_para2", "Wisconsin_Legislature_para1", "William_Huntington_Kirkpatrick_para6",
        "Nevada_Legislature_para7", "William_Sebring_Kirkpatrick_para7", "Nevada_Legislature_para5",
        "John_Roger_Kirkpatrick_Scott_para1", "Nevada_Legislature_para6", "Franklin_Coslett_para6", "Hawaii_State_Legislature_para0"
      ]
    },
    {
      index: 5,
      id: "5ab985eb554299131ca42360",
      question: "Between Greyia and Calibanus, which genus contains more species?",
      gold_answer: "Greyia",
      monolithic: {
        predicted_answer: "Greyia contains more species.",
        retrieval_ms: 322.1,
        generation_ms: 2237.8,
        scores: { faithfulness: 1.0, answer_relevancy: 0.864, context_precision: 0.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Greyia contains more species.",
        retrieval_ms: 1113.4,
        generation_ms: 7059.8,
        scores: { faithfulness: 1.0, answer_relevancy: 0.864, context_precision: 0.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Greyia_para5", "Grasswren_para7", "Calibanus_para8", "Calyptocephalellidae_para1",
        "List_of_Utricularia_species_para8", "Calyptocephalellidae_para4", "Caesalpinioideae_para9",
        "Calyptocephalellidae_para9", "Dactylis_para6", "Calyptocephalellidae_para5"
      ]
    },
    {
      index: 6,
      id: "5a8753d95542994846c1cd63",
      question: "Did John Updike and Tom Clancy both publish more than 15 bestselling novels?",
      gold_answer: "yes",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 436.8,
        generation_ms: 2240.1,
        scores: { faithfulness: 1.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1122.8,
        generation_ms: 6536.5,
        scores: { faithfulness: 1.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "John_Updike_para2", "John_Updike_para1", "Tom_Clancy_para5", "Tom_Clancy_para3",
        "Jack_Ryan_(character)_para8", "Tom_Clancy's_Net_Force_Explorers:_Deathworld_para1",
        "Tom_Clancy's_Net_Force_Explorers:_High_Wire_para2", "Philip_José_Farmer_bibliography_para6",
        "Tom_Clancy's_Net_Force_Explorers:_Duel_Identity_para3", "Patrick_K._O'Donnell_para6"
      ]
    },
    {
      index: 7,
      id: "5ab97d0a5542996be202051e",
      question: "Who was hung for assisting the attempted surrender of a defector from the American Continental Army to the British Army?",
      gold_answer: "John André",
      monolithic: {
        predicted_answer: "John André.",
        retrieval_ms: 629.6,
        generation_ms: 10031.9,
        scores: { faithfulness: 1.0, answer_relevancy: 0.515, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "John André.",
        retrieval_ms: 1704.8,
        generation_ms: 11025.4,
        scores: { faithfulness: 1.0, answer_relevancy: 0.624, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "John_André_para6", "Benjamin_Lincoln_para0", "Benedict_Arnold_para3", "Siege_of_Yorktown_para2",
        "Sir_John_Wedderburn,_5th_Baronet_of_Blackness_para2", "List_of_Continental_Army_units_para8",
        "Charles_White_Whittlesey_para3", "South_German_Offensive_para0", "William_Farrar_Smith_para1", "South_German_Offensive_para3"
      ]
    },
    {
      index: 8,
      id: "5adef1b35542993a75d263af",
      question: "Which Mexican and American film actress is Ethel Houbiers French voice of?",
      gold_answer: "Salma Hayek Pinault",
      monolithic: {
        predicted_answer: "Salma Hayek.",
        retrieval_ms: 438.2,
        generation_ms: 915.5,
        scores: { faithfulness: 0.5, answer_relevancy: 0.127, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Salma Hayek.",
        retrieval_ms: 1093.2,
        generation_ms: 4448.0,
        scores: { faithfulness: 1.0, answer_relevancy: 0.285, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Ethel_Houbiers_para2", "Bernard_Alane_para7", "Dorothée_Jemma_para4", "Pascal_Mazzotti_para8",
        "Pascal_Mazzotti_para0", "Lumi_Cavazos_para8", "Ethel_Rojo_para7", "Roger_Carel_para5",
        "Rozie_Curtis_para8", "Virginie_Ledieu_para8"
      ]
    },
    {
      index: 9,
      id: "5a73b55855429978a71e9086",
      question: "Which major international airport in south-east England ranks as the 8th busiest airport in Europe and replaced Croydon Airport?",
      gold_answer: "Gatwick Airport",
      monolithic: {
        predicted_answer: "Gatwick Airport (London Gatwick).",
        retrieval_ms: 657.8,
        generation_ms: 3377.0,
        scores: { faithfulness: 1.0, answer_relevancy: 0.593, context_precision: 0.833, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Gatwick Airport (London Gatwick).",
        retrieval_ms: 1564.2,
        generation_ms: 10236.0,
        scores: { faithfulness: 1.0, answer_relevancy: 0.612, context_precision: 0.833, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Gatwick_Airport_para1", "Gatwick_Airport_para9", "Heathrow_Airport_para8", "Heathrow_Airport_para9",
        "Grantley_Adams_International_Airport_para0", "Croydon_Airport_para5", "Grantley_Adams_International_Airport_para8",
        "Grantley_Adams_International_Airport_para1", "Grantley_Adams_International_Airport_para4", "International_air_travel_from_the_United_Kingdom_para3"
      ]
    },
    {
      index: 10,
      id: "5ae80919554299540e5a56f6",
      question: "Isabella Kelly was born at a ruined castle characterized as one of the most isolated fortifications in Britain by who?",
      gold_answer: "The Changing Scottish Landscape",
      monolithic: {
        predicted_answer: "It was described as such by the 1991 book The Changing Scottish Landscape.",
        retrieval_ms: 639.7,
        generation_ms: 2562.4,
        scores: { faithfulness: 1.0, answer_relevancy: 0.150, context_precision: 0.500, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "It was described that way by the 1991 publication The Changing Scottish Landscape.",
        retrieval_ms: 1695.8,
        generation_ms: 7688.1,
        scores: { faithfulness: 1.0, answer_relevancy: 0.0, context_precision: 0.500, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Castillo_de_Tabernas_para5", "Cairnburgh_Castle_para7", "Isabella_Kelly_para8", "Kildrummy_Castle_para3",
        "Edinburgh_Castle_para0", "Inch_Castle_para4", "Inverallochy_Castle_para4", "Comrie_Castle_para7",
        "Tantallon_Castle_para3", "Schauenberg_para9"
      ]
    },
    {
      index: 11,
      id: "5ae361605542994393b9e69b",
      question: "The facility where Robert Croft worked as a navy instructor was also known as?",
      gold_answer: "\"Home of the Submarine Force\"",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 506.7,
        generation_ms: 14284.8,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1259.5,
        generation_ms: 6386.5,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Robert_Croft_(diver)_para7", "Otto_Ege_para6", "RAF_Croft_para4", "Music_Instructor_para1",
        "USS_Shannon_para0", "Naval_Consolidated_Brig,_Miramar_para0", "Officer_Training_Command_Newport_para2",
        "Norfolk_Naval_Shipyard_para4", "Peter_Hudson_(Royal_Navy_officer)_para3", "Davis_Avenue_Recreation_Center_para9"
      ]
    },
    {
      index: 12,
      id: "5ac3a76e554299741d48a2be",
      question: "Who released the song \"With or Without You\" first, Jai McDowall or U2?",
      gold_answer: "U2",
      monolithic: {
        predicted_answer: "U2 released the song \"With or Without You\" first.",
        retrieval_ms: 453.0,
        generation_ms: 3973.1,
        scores: { faithfulness: 1.0, answer_relevancy: 0.750, context_precision: 0.556, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "U2 released the song \"With or Without You\" first.",
        retrieval_ms: 1098.9,
        generation_ms: 6974.7,
        scores: { faithfulness: 1.0, answer_relevancy: 0.769, context_precision: 0.556, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Believe_(Jai_McDowall_album)_para7", "With_or_Without_You_para2", "With_or_Without_You_para6",
        "Mercy_(Steve_Jones_album)_para6", "Lonestar_discography_para5", "LMC_(British_band)_para3",
        "Join_Hands_para0", "Brothers_of_the_Road_para8", "Gravity_(Embrace_song)_para3", "Always_You_(Jennifer_Paige_song)_para8"
      ]
    },
    {
      index: 13,
      id: "5a7557d75542992d0ec05f68",
      question: "Which \"Roseanne\" star is in Scream 2?",
      gold_answer: "Laurie Metcalf",
      monolithic: {
        predicted_answer: "",
        retrieval_ms: 290.2,
        generation_ms: 8804.7,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.500, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 772.6,
        generation_ms: 8238.5,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.500, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Star_Plus_(UK_and_Ireland)_para7", "Scream_2_para6", "Johnny_Galecki_para2", "Bobby_Gaylor_para4",
        "Johnny_Galecki_para4", "Glenn_Quinn_para9", "John_Goodman_para0", "Roseanne_Roseannadanna_para4",
        "Union_(band)_para9", "Sidney_Prescott_para5"
      ]
    },
    {
      index: 14,
      id: "5a8d12ca5542994ba4e3dbe2",
      question: "In what city is the company that Fastjet Tanzania was originally founded as a part of prior to rebranding based?",
      gold_answer: "Nairobi, Kenya",
      monolithic: {
        predicted_answer: "Dar es Salaam.",
        retrieval_ms: 701.2,
        generation_ms: 10557.6,
        scores: { faithfulness: 0.0, answer_relevancy: 0.491, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "Dar es Salaam.",
        retrieval_ms: 1704.3,
        generation_ms: 9597.8,
        scores: { faithfulness: 0.0, answer_relevancy: 0.491, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Fastjet_Tanzania_para2", "Hall_Park,_Oklahoma_para3", "Restaurant_Brands_International_para0",
        "Shop_Direct_para0", "Namibia_Breweries_Limited_para1", "Garrard_County,_Kentucky_para8",
        "Netball_in_Tanzania_para7", "National_City_Lines_para7", "Learjet_para0", "Alfred_Hatch_Place_at_Arcola_para6"
      ]
    },
    {
      index: 15,
      id: "5ab7f6d35542993667794070",
      question: "What is the name of the Australian specialist electronic music magazine that features avant-rock, experimental sound art, and experimental hip hop?",
      gold_answer: "Cyclic Defrost",
      monolithic: {
        predicted_answer: "Cyclic Defrost.",
        retrieval_ms: 625.2,
        generation_ms: 8486.5,
        scores: { faithfulness: 0.0, answer_relevancy: 0.035, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Cyclic Defrost.",
        retrieval_ms: 1791.4,
        generation_ms: 8330.5,
        scores: { faithfulness: 0.0, answer_relevancy: 0.035, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Cyclic_Defrost_para6", "Music_Connection_para9", "Petri_Kuljuntausta_para7", "Music_Connection_para6",
        "List_of_sound_artists_para2", "ROCKRGRL_para2", "Experimental_hip_hop_para4", "Blender_(magazine)_para7",
        "Sixtoo_para1", "M_Music_&_Musicians_para9"
      ]
    },
    {
      index: 16,
      id: "5ac3d9135542995c82c4ac4c",
      question: "What is the population of the city that Munsonville is in the northwest corner of?",
      gold_answer: "729 at the 2010 census",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1152.0,
        generation_ms: 6407.2,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1399.3,
        generation_ms: 6282.7,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Munsonville,_New_Hampshire_para9", "Dade_County,_Georgia_para2", "Buchanan_Township,_Atchison_County,_Missouri_para1",
        "Murwillumbah_para9", "Munson_Valley_Historic_District_para9", "West_Sand_Lake,_New_York_para0",
        "Morrisville,_North_Carolina_para6", "Muskegon,_Michigan_para3", "Munson_Diner_para6", "Fairhope,_Fayette_County,_Pennsylvania_para8"
      ]
    },
    {
      index: 17,
      id: "5ae129115542990adbacf722",
      question: "A Pair of Brown Eyes and Wild Mountain Thyme is based from what artists song?",
      gold_answer: "Francis McPeake",
      monolithic: {
        predicted_answer: "",
        retrieval_ms: 1305.1,
        generation_ms: 9478.0,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.393, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "",
        retrieval_ms: 1676.2,
        generation_ms: 8433.3,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.393, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Brown_Eyes_(band)_para5", "A_Pair_of_Brown_Eyes_para1", "Brown_Mountain_Lights_para6",
        "Volcanic_City_para7", "Beautiful_Brown_Eyes_para9", "Snake_eyes_para9", "A_Spanner_in_the_Works_para3",
        "Anything's_Wild_para7", "The_Monarch_(production_team)_para8", "Brown_Eyes_(song)_para6"
      ]
    },
    {
      index: 18,
      id: "5a86769c5542994775f60776",
      question: "Armageddon in Retrospect was written by the author who was best known for what 1969 satire novel?",
      gold_answer: "Slaughterhouse-Five",
      monolithic: {
        predicted_answer: "",
        retrieval_ms: 1585.5,
        generation_ms: 7982.4,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 1.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "",
        retrieval_ms: 4004.7,
        generation_ms: 4631.0,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Armageddon_in_Retrospect_para9", "Vincent_Musetto_para3", "Armageddon_(Swedish_band)_para4",
        "Armageddon_(2004)_para8", "The_Man_with_the_Golden_Arm_(novel)_para7", "Lettice_and_Lovage_para1",
        "Nelson_Algren_para5", "O_Cangaceiro_para4", "Avant-Pop:_Fiction_for_a_Daydream_Nation_para7", "The_Scarlet_Empire_para9"
      ]
    },
    {
      index: 19,
      id: "5ab31864554299233954ff06",
      question: "What class of instrument does Apatim Majumdar play?",
      gold_answer: "strings",
      monolithic: {
        predicted_answer: "Apratim Majumdar plays the sarod, which is a stringed instrument.",
        retrieval_ms: 866.9,
        generation_ms: 9909.8,
        scores: { faithfulness: 1.0, answer_relevancy: 0.887, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Apratim Majumdar plays the sarod, which is a stringed instrument.",
        retrieval_ms: 906.5,
        generation_ms: 10566.6,
        scores: { faithfulness: 1.0, answer_relevancy: 0.887, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Apratim_Majumdar_para0", "Santoor_para7", "Experimental_musical_instrument_para5",
        "Chander_Bari_para2", "Magnetometer_(Juno)_para3", "Umuduri_para2", "Magnetometer_(Juno)_para2",
        "Sarod_para1", "Asset_classes_para8", "Malunga_para4"
      ]
    },
    {
      index: 20,
      id: "5a7b1023554299042af8f6c2",
      question: "Which movie did Disney produce first, The Many Adventures of Winnie the Pooh or Ride a Wild Pony?",
      gold_answer: "Ride a Wild Pony",
      monolithic: {
        predicted_answer: "Ride a Wild Pony.",
        retrieval_ms: 1746.5,
        generation_ms: 8443.9,
        scores: { faithfulness: 1.0, answer_relevancy: 0.050, context_precision: 0.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Ride a Wild Pony was produced first.",
        retrieval_ms: 1708.3,
        generation_ms: 9155.2,
        scores: { faithfulness: 0.0, answer_relevancy: 0.432, context_precision: 0.167, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "The_Many_Adventures_of_Winnie_the_Pooh_para5", "Winnie_the_Pooh_and_the_Blustery_Day_para6",
        "Winnie_the_Pooh_and_the_Blustery_Day_para7", "The_Wonderful_Thing_About_Tiggers_para4",
        "The_Wonderful_Thing_About_Tiggers_para9", "Ride_a_Wild_Pony_para6", "The_Wonderful_Thing_About_Tiggers_para6",
        "Ride_a_Wild_Pony_para8", "A_Valentine_for_You_para7", "Heffalumps_and_Woozles_para1"
      ]
    },
    {
      index: 21,
      id: "5ac19f405542991316484b5b",
      question: "Pandikona and Berger Blanc Suisse are both what kinds of animals?",
      gold_answer: "dogs",
      monolithic: {
        predicted_answer: "Both are breeds of dogs.",
        retrieval_ms: 1202.1,
        generation_ms: 6843.7,
        scores: { faithfulness: 1.0, answer_relevancy: 0.435, context_precision: 0.317, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Both are dogs.",
        retrieval_ms: 1065.0,
        generation_ms: 6260.0,
        scores: { faithfulness: 1.0, answer_relevancy: 0.539, context_precision: 0.310, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Berger_Blanc_Suisse_para4", "Berger_Blanc_Suisse_para9", "Pandikona_para1", "Piranha:_Wolf_In_The_Water_para5",
        "List_of_Scaevola_species_para8", "Tentacle_para7", "Chien_Français_Blanc_et_Noir_para6", "Peter_L._Berger_para4",
        "Chien_Français_Blanc_et_Orange_para2", "Rainbow_Bridge_(film)_para7"
      ]
    },
    {
      index: 22,
      id: "5ae382ee5542992e3233c430",
      question: "The Prussian General Carl von Clausewitz is associated with what type of realism?",
      gold_answer: "Modern thinkers associated with classical realism are Carl von Clausewitz",
      monolithic: {
        predicted_answer: "Classical realism.",
        retrieval_ms: 1180.3,
        generation_ms: 8135.2,
        scores: { faithfulness: 1.0, answer_relevancy: 0.412, context_precision: 0.167, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Classical realism.",
        retrieval_ms: 1191.6,
        generation_ms: 8142.2,
        scores: { faithfulness: 1.0, answer_relevancy: 0.319, context_precision: 0.167, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Carl_von_Clausewitz_para8", "Absolute_war_para3", "On_War_para9", "August_Otto_Rühle_von_Lilienstern_para7",
        "Christopher_Bassford_para6", "Classical_realism_(international_relations)_para1", "Charles_Roberts_Awards_Hall_of_Fame_para0",
        "Battle_of_Wavre_para5", "Ernst_Ludwig_von_Gerlach_para4", "Panzer_Division_Clausewitz_para2"
      ]
    },
    {
      index: 23,
      id: "5adcdf455542994ed6169c11",
      question: "What where both Hawker Hurricane and No. 1455 Flight apart of?",
      gold_answer: "Royal Air Force",
      monolithic: {
        predicted_answer: "Both the Hawker Hurricane and No. 1455 Flight were part of the Royal Air Force.",
        retrieval_ms: 1068.0,
        generation_ms: 5764.7,
        scores: { faithfulness: 1.0, answer_relevancy: 0.869, context_precision: 0.861, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Both the Hawker Hurricane and No. 1455 Flight were part of the Royal Air Force.",
        retrieval_ms: 1031.9,
        generation_ms: 6264.3,
        scores: { faithfulness: 1.0, answer_relevancy: 0.869, context_precision: 0.861, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "No._1455_Flight_RAF_para5", "No._1300_Flight_RAF_para4", "Hawker_Hurricane_para8", "Sindlinger_Hawker_Hurricane_para7",
        "Hawker_Typhoon_para2", "Flying_Legend_Hawker_Hurricane_Replica_para1", "Hawker_Tornado_para6", "Tenerife_airport_disaster_para4",
        "Battle_of_Britain_Memorial_Flight_para9", "Hawker_Hurricane_in_Yugoslav_service_para3"
      ]
    },
    {
      index: 24,
      id: "5ac35a7e554299741d48a257",
      question: "What language did the ethnic group which Torstein Ellingsen was its drumer speaks?",
      gold_answer: "Norwegian language",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1362.7,
        generation_ms: 7163.2,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.833, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1114.4,
        generation_ms: 7306.3,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.833, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Torstein_Ellingsen_para9", "Rotvälsk_para2", "The_Real_Thing_(Norwegian_band)_para0", "Tugurt_language_para8",
        "Fezara_people_para6", "Fezara_people_para2", "Siwi_people_para5", "Fezara_people_para1", "Berbers_para2", "North_Group_para9"
      ]
    },
    {
      index: 25,
      id: "5ab2659e554299340b5254b2",
      question: "From March 631 to April 631, Farrukhzad Khosrau V was the king of an empire that succeeded which empire?",
      gold_answer: "the Parthian Empire",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1739.1,
        generation_ms: 9513.4,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1711.8,
        generation_ms: 1021.7,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Farrukhzad_Khosrow_V_para2", "Hormizd_VI_para9", "Kassapa_III_of_Anuradhapura_para0", "Khosrow_IV_para1",
        "Duchy_of_Thuringia_para0", "Narathihapate_para3", "Samo's_Empire_para7", "House_of_Theemuge_para1",
        "Ashur-etil-ilani_para5", "Kashyapa_I_of_Anuradhapura_para9"
      ]
    },
    {
      index: 26,
      id: "5adcbb815542994ed6169bde",
      question: "Beer Wars covers the differences between large corporate breweries, and small breweries, such as what brewery that is headquartered in Escondido, california?",
      gold_answer: "Stone Brewing",
      monolithic: {
        predicted_answer: "Stone Brewing.",
        retrieval_ms: 1975.7,
        generation_ms: 7079.1,
        scores: { faithfulness: 1.0, answer_relevancy: 0.608, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Stone Brewing.",
        retrieval_ms: 2000.4,
        generation_ms: 731.9,
        scores: { faithfulness: 1.0, answer_relevancy: 0.576, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Beer_Wars_para9", "Stone_Brewing_Co._para8", "List_of_breweries_in_California_para2", "Microbrewery_para4",
        "Brewery_para2", "List_of_microbreweries_para7", "Boulevard_Brewing_Company_para0", "Tusker_(beer)_para5",
        "Beer_by_region_para6", "Lion_Red_para3"
      ]
    },
    {
      index: 27,
      id: "5a74f8bd5542993748c8976a",
      question: "Which head coach has led their team for a longer period of time, Tim Cluess or Steve Prohm?",
      gold_answer: "Tim Cluess",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1643.8,
        generation_ms: 4940.1,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1693.6,
        generation_ms: 623.2,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.0, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "2007_South_Carolina_Gamecocks_football_team_para4", "David_Hobbs_(basketball)_para4", "2009_Oregon_Ducks_football_team_para7",
        "Steve_Spurrier_para7", "2009_Oregon_Ducks_football_team_para4", "Barry_Alvarez_para4", "2017_Texas_Tech_Red_Raiders_baseball_team_para0",
        "Steve_Payne_(basketball)_para3", "2016_Texas_Tech_Red_Raiders_baseball_team_para5", "Mike_Leach_(American_football_coach)_para6"
      ]
    },
    {
      index: 28,
      id: "5ade25ed5542997c77aded70",
      question: "During what war were the Russia-United Kingdom relations in a state of rivalry after the abdication of Emperor Nicholas II?",
      gold_answer: "the Cold War (1947–91)",
      monolithic: {
        predicted_answer: "",
        retrieval_ms: 1637.9,
        generation_ms: 13511.6,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 1.0, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "The rivalry occurred during the Cold War (1947–1991).",
        retrieval_ms: 1766.6,
        generation_ms: 11748.6,
        scores: { faithfulness: 1.0, answer_relevancy: 0.442, context_precision: 1.0, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Russia–United_Kingdom_relations_para4", "Charles_Sydney_Gibbes_para1", "Charles_Sydney_Gibbes_para2", "Egypt–Italy_relations_para2",
        "Nicholas_II_of_Russia_para6", "Russian_Revolution_para3", "Nicholas_I_of_Russia_para1", "Russian_Revolution_para8",
        "Nicholas_I_of_Russia_para3", "Grand_Duke_Sergei_Alexandrovich_of_Russia_para1"
      ]
    },
    {
      index: 29,
      id: "5a78ae8b5542990784727730",
      question: "How far from Sacramento is the flight school in Atwater?",
      gold_answer: "about 115 miles (185 km)",
      monolithic: {
        predicted_answer: "Approximately 115 miles (about 185 km) south of Sacramento.",
        retrieval_ms: 968.2,
        generation_ms: 7593.0,
        scores: { faithfulness: 1.0, answer_relevancy: 0.637, context_precision: 0.100, context_recall: 1.0 }
      },
      patternrag: {
        predicted_answer: "Approximately 115 miles (about 185 km) south of Sacramento.",
        retrieval_ms: 953.7,
        generation_ms: 7219.4,
        scores: { faithfulness: 1.0, answer_relevancy: 0.620, context_precision: 0.100, context_recall: 1.0 }
      },
      retrieved_doc_ids: [
        "Pacific_Flying_Club_para1", "Sacramento_Deep_Water_Ship_Channel_para3", "American_Flyers_(flight_school)_para5",
        "Huffman_Aviation_para9", "Sierra_Academy_of_Aeronautics_para6", "North_American_Institute_of_Aviation_para2",
        "Singapore_Flying_College_para1", "Lemoore_Army_Air_Field_para9", "List_of_U.S._Air_Force_Test_Pilot_School_alumni_para0", "Castle_Air_Force_Base_para0"
      ]
    },
    {
      index: 30,
      id: "5ae12a4b55429920d52342df",
      question: "Baraki Barak District is situated in the western part of a province whose capital is what?",
      gold_answer: "Puli Alam",
      monolithic: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1373.1,
        generation_ms: 7701.9,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.250, context_recall: 0.0 }
      },
      patternrag: {
        predicted_answer: "I don't know.",
        retrieval_ms: 1319.6,
        generation_ms: 7183.7,
        scores: { faithfulness: 0.0, answer_relevancy: 0.0, context_precision: 0.250, context_recall: 0.0 }
      },
      retrieved_doc_ids: [
        "Baraki_Barak_District_para6", "Baraki_Rajan_para4", "Baraki_Barak_para8", "Baraki_Barak_airstrike_para3",
        "Nahal_Barak_para8", "Bharatpur_district_para1", "Camptown_(country_subdivision)_para7", "Lishu_District_para0",
        "Muzaffarnagar_para7", "Yuexiu_District_para7"
      ]
    }
  ]
};
