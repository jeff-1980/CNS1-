# Structured card schema (one JSON object per paper, JSONL)

{
 "id": "W...",                          // copy from input
 "object": "...",                       // research object, English noun phrase (e.g. "wind turbine drivetrain bearings")
 "problem": "...",                      // the one question the paper answers, <= 25 words
 "mechanism_or_method": "...",          // the key mechanism / method / design idea, <= 25 words
 "data": {
   "type": "lab_experiment|fabrication|field_operational|public_dataset|simulation|theory|survey|mixed",
   "public": "yes|no|unclear",          // is the main evidence data publicly available (as stated or clearly implied)
   "real_world": "yes|no",              // from real deployed systems / field / patients, not bench or synthetic
   "longitudinal_years": null|number,   // span of real-world data if any
   "named_datasets": ["..."]            // named public datasets/sources mentioned, else []
 },
 "evidence": ["fabrication","wet_lab","bench_experiment","field_trial","simulation","theory","data_analysis"],  // all that apply
 "pure_computational": "yes|no",        // could the whole paper be done with public data + one GPU, no new hardware/wet lab/fabrication
 "requires_hardware": "yes|no",         // needs a fabricated device, test rig, robot, or new sensing campaign
 "claim_level": "T1|T2|T3|T4",          // T1 overturns/explains a long-standing puzzle (mechanism); T2 first demonstration of a capability/method validated on real data; T3 improvement on known approach; T4 dataset/resource/benchmark
 "lever": "none|meta_research|large_scale_operational_data|model_plus_real_event|pure_method_theory|new_dataset",  // only for pure_computational=yes, else "none"
 "mech_relevance": "core|adjacent|tagged_only",  // core = mechanical engineering question; adjacent = materials/energy/robotics with mechanical content; tagged_only = chemistry/bio etc. with no mechanical question
 "limitation": "...",                   // main stated or evident limitation, <= 20 words
 "decision_relevance": "...",           // what engineering decision the result changes, or "none", <= 15 words
 "keywords": ["...", "...", "...", "...", "..."]   // 5 lower-case English keywords
}
