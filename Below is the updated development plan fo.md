Below is the updated development plan for the High-Performance Name Matching Application, replacing Docker with Podman for containerization while retaining the comprehensive structure, tasks, deliverables, and KPIs from the previous plan. Podman is a daemonless, rootless container engine that provides a Docker-compatible CLI, ensuring seamless integration with Kubernetes and minimal changes to the deployment strategy. The plan remains tailored to Filipino identity data, supporting table-to-table and CSV-to-table matching with MySQL, Rust, and GPU acceleration.


# Enhanced Name Matching Application Development Plan with Podman

## 1. Executive Summary
This plan details the development of a high-performance name matching application for Filipino identity data, enabling matching between two MySQL tables with identical schemas (hh_id, first_name, middle_name_last_name, birthdate, province_name, city_name, barangay_name) or a CSV input that creates a temporary table matched against a second MySQL table. The solution leverages MySQL for data management, Rust for core logic and parsing, and GPU acceleration for fuzzy matching. Enhancements include flexible input handling, optimized blocking, advanced parsing, GPU pipeline optimization, robust security, and an extended API. Podman replaces Docker for containerization, offering daemonless, rootless operation with Kubernetes integration. The plan is structured in three phases with detailed tasks, deliverables, and KPIs to ensure accuracy, throughput, scalability, and security.

## 2. System Overview
### 2.1. Objectives
- **Accuracy**: Achieve >0.9 F1-score for Filipino names, handling complex middle_name_last_name fields and cultural nuances.
- **Performance**: Process >10,000 record pairs/second for table-to-table matching and <1min for CSV batches of 10,000 records.
- **Scalability**: Support petabyte-scale datasets with efficient blocking and distributed processing.
- **Flexibility**: Handle same-format table matching and CSV-to-table matching seamlessly.
- **Security**: Ensure robust PII protection across all components.

### 2.2. Technology Stack
- **MySQL**: Data storage, indexing, and candidate pre-filtering (utf8mb4, B-Tree, FULLTEXT ngram indexes).
- **Rust**: Core logic, parsing, preprocessing, and GPU orchestration (crates: `sqlx`, `csv`, `cudarc`, `unicode-normalization`, `rayon`, `tracing`).
- **GPU**: CUDA/OpenCL for fuzzy matching (Jaro-Winkler, Levenshtein) and optional blocking.
- **API**: RESTful interface for table and CSV matching, built with `actix-web` or `axum`.
- **Deployment**: Podman for containerization, Kubernetes for orchestration, ensuring daemonless and rootless operation.

### 2.3. Data Flow
1. **Input**: Receive two table names or a CSV file and table name via API.
2. **CSV Import (if applicable)**: Parse CSV in Rust, create temporary MySQL table, insert records with indexes.
3. **Candidate Generation**: Query MySQL for candidate pairs using geographic, birthdate, and phonetic blocking.
4. **Preprocessing**: Parse and normalize records in Rust, generate features (standardized names, phonetic codes, n-grams).
5. **GPU Matching**: Batch pairs to GPU for coarse (e.g., Jaccard on n-grams) and fine-grained (e.g., Jaro-Winkler) matching.
6. **Scoring**: Aggregate scores in Rust, classify matches (Match/Non-Match/Manual Review), return results via API or store in MySQL.

## 3. Development Phases

### Phase 1: Core CPU Matching Engine & Data Foundation (Months 1-6)
**Objective**: Build a functional baseline with robust parsing, CPU-based matching, and flexible input handling.

**Tasks**:
1. **MySQL Schema Design**:
   - Finalize schema: `hh_id` (BIGINT UNSIGNED, PRIMARY KEY), `first_name` (VARCHAR(100)), `middle_name_last_name` (VARCHAR(200)), `birthdate` (DATE), `province_name` (VARCHAR(100)), `city_name` (VARCHAR(100)), `barangay_name` (VARCHAR(100)).
   - Use utf8mb4_unicode_ci for Unicode support.
   - Create indexes: B-Tree on `hh_id`, `birthdate`, `(province_name, city_name, barangay_name)`; optional FULLTEXT ngram on `first_name`, `middle_name_last_name` (ngram_token_size=2).
2. **CSV Input Handling**:
   - Implement CSV parsing with `csv` crate, mapping columns to schema via TOML/YAML config.
   - Validate CSV data (e.g., date formats, length limits).
   - Create temporary MySQL table with `sqlx`, mirroring main schema, and insert records in batches.
3. **Rust Orchestration Service**:
   - Develop service to handle table-to-table and CSV-to-table inputs.
   - Use `sqlx` for async MySQL queries, supporting connection pooling.
   - Implement query construction for candidate generation (e.g., `WHERE province_name = ? AND ABS(DATEDIFF(birthdate, ?)) <= 7`).
4. **Parsing and Preprocessing**:
   - Build rule-based parser for `middle_name_last_name`, recognizing Filipino patterns (e.g., “Dela Cruz” as single surname).
   - Integrate dictionary of Filipino nicknames and surname variants.
   - Implement Unicode normalization (NFKC with `unicode-normalization`), case folding, whitespace trimming, and special character handling (e.g., retain hyphens in “Santos-Dizon”).
   - Standardize birthdates to ISO 8601 with `chrono`.
   - Standardize geographic fields against PSGC gazetteer.
5. **CPU-Based Matching**:
   - Implement Jaro-Winkler, Soundex, and Jaccard Index using `fuzzy_name_match` crate.
   - Apply Jaro-Winkler to `first_name` and parsed `middle_name_last_name` components, Soundex for phonetic checks, Jaccard for token order variations.
   - Perform exact matching on standardized `birthdate` and geographic fields.
6. **Scoring and Decisioning**:
   - Develop weighted scoring model (e.g., 40% names, 30% birthdate, 30% geography).
   - Set thresholds for Match (>0.85), Non-Match (<0.65), Manual Review (0.65-0.85).
7. **Testing Framework**:
   - Curate Filipino name dataset with verified matches/non-matches.
   - Benchmark accuracy (precision, recall, F1-score) and throughput.
8. **Logging and Monitoring**:
   - Use `tracing` for structured logging (e.g., CSV import, query performance).
   - Track metrics (e.g., records processed, error rates) with `prometheus`.
9. **Podman Setup**:
   - Create Podman containers for Rust application and MySQL (development/testing).
   - Use Podman’s Docker-compatible CLI to build images with CUDA/OpenCL dependencies.
   - Test rootless execution to ensure security.

**Deliverables**:
- MySQL schema and indexes for main and temporary tables.
- Rust service for CSV import, table matching, parsing, and CPU-based matching.
- Initial Filipino name dictionary and preprocessing routines.
- Podman container definitions for Rust and MySQL.
- Benchmark report with accuracy (>0.8 F1-score) and throughput (>1,000 pairs/second).

**KPIs**:
- **Accuracy**: F1-score >0.8 on test dataset.
- **Throughput**: >1,000 pairs/second on CPU.
- **Error Rate**: <0.1% for CSV imports and table creation.
- **Latency**: P95 <2s for single-record queries.
- **Podman Stability**: 100% successful container builds and rootless execution.

### Phase 2: GPU Acceleration and Performance Optimization (Months 7-12)
**Objective**: Accelerate matching with GPU kernels and optimize for large-scale table comparisons.

**Tasks**:
1. **Profiling**:
   - Profile Phase 1 application with `cargo flamegraph` to identify bottlenecks (e.g., Jaro-Winkler on large candidate sets).
2. **GPU Kernel Development**:
   - Develop CUDA/OpenCL kernels for Jaro-Winkler (pairwise character matching, transposition counting) and Levenshtein (anti-diagonal DP matrix) using `cudarc` or `opencl3`.
   - Implement coarse pass (Jaccard on n-grams) for rapid filtering.
   - Use shared memory for DP matrix rows, coalesced global memory for strings.
3. **Data Marshalling**:
   - Batch pairs by similar string lengths to minimize padding.
   - Use pinned host memory and CUDA streams for asynchronous transfers.
4. **Integration**:
   - Integrate GPU kernels into Rust via FFI, ensuring efficient data serialization (e.g., `Vec<repr(C) struct>`).
   - Pipeline coarse and fine passes: coarse pass filters to <10% of pairs, fine pass applies Jaro-Winkler/Levenshtein.
5. **Blocking Optimization**:
   - Enhance MySQL blocking with phonetic keys (Soundex on parsed last names).
   - Prototype GPU-accelerated blocking (e.g., n-gram profile computation) inspired by HyperBlocker.
6. **Performance Tuning**:
   - Use Nsight Compute to tune kernel block sizes, shared memory, and occupancy.
   - Optimize query plans with MySQL `EXPLAIN` to avoid full table scans.
   - Benchmark GPU vs. CPU performance, targeting >10,000 pairs/second.
7. **Security Enhancements**:
   - Implement TLS for MySQL and API connections.
   - Use MySQL TDE for temporary tables.
   - Sanitize CSV inputs and zero out GPU memory buffers post-processing.
8. **Podman Integration**:
   - Update Podman containers to include GPU dependencies (NVIDIA CUDA drivers, OpenCL runtimes).
   - Test Podman’s `--gpus` flag for GPU access in rootless mode.
   - Ensure compatibility with Kubernetes via Podman-generated pod specs.

**Deliverables**:
- GPU-accelerated matching pipeline with coarse and fine passes.
- Optimized blocking strategies for table-to-table matching.
- Security measures for PII handling.
- Updated Podman containers with GPU support.
- Benchmark report showing >10,000 pairs/second and <1min for 10,000-record CSV batches.

**KPIs**:
- **Throughput**: >10,000 pairs/second on GPU.
- **Latency**: P95 <1s for single queries, <1min for 10,000-record batches.
- **Accuracy**: F1-score >0.85.
- **Scalability**: No significant performance degradation with 10x data increase.
- **Podman GPU Access**: 100% successful GPU kernel execution in containers.

### Phase 3: Advanced Features, Scalability, and Production Hardening (Months 13-18)
**Objective**: Add advanced features, ensure scalability, and prepare for production.

**Tasks**:
1. **Manual Review UI and XAI**:
   - Build web-based UI with React for reviewing ambiguous matches.
   - Display field scores, rule activations, and feature importance (e.g., SHAP for scoring model).
   - Integrate with API for fetching review tasks.
2. **API Extension**:
   - Implement `/match/table` and `/match/batch/{job_id}` endpoints.
   - Support configuration (thresholds, max results) and HATEOAS.
3. **Advanced Blocking**:
   - Implement Sorted Neighborhood Method or LSH in Rust.
   - Scale GPU blocking for petabyte datasets.
4. **Deep Learning Exploration**:
   - Prototype Transformer-based similarity scoring (e.g., Sentence-BERT) via ONNX Runtime.
   - Evaluate accuracy vs. traditional algorithms.
5. **Scalability**:
   - Shard MySQL tables for petabyte-scale data.
   - Deploy Rust services as stateless Kubernetes pods, using Podman to generate pod specs.
6. **Resilience**:
   - Implement retries, circuit breakers, and DLQs for failed records.
   - Monitor with Prometheus/Grafana, set alerts for anomalies.
7. **Security**:
   - Enforce RBAC for MySQL, Rust, and UI access.
   - Explore PPRL techniques (e.g., Bloom filters) for future privacy needs.
8. **Deployment**:
   - Finalize Podman containers with CUDA/OpenCL dependencies.
   - Configure Kubernetes for GPU allocation and scaling, using Podman’s `podman generate kube` for pod specs.
   - Externalize config with TOML/YAML and secrets via environment variables.
   - Test rootless Podman execution in production-like environment.

**Deliverables**:
- Production-ready application with UI, XAI, and extended API.
- Scalable architecture handling petabyte-scale data.
- Comprehensive security and resilience measures.
- Podman-based Kubernetes deployment scripts.
- Final benchmark report with all KPIs met.

**KPIs**:
- **Accuracy**: F1-score >0.9.
- **Throughput**: >15,000 pairs/second.
- **Latency**: P95 <0.5s for queries, <30s for 10,000-record batches.
- **Scalability**: Linear performance scaling to 100x data.
- **Cost**: TCO competitive with commercial solutions at scale.
- **Usability**: >90% user satisfaction with UI/XAI.
- **Podman Reliability**: Zero downtime due to container issues in production.

## 4. Recommendations
1. **Prioritize Parsing**: Invest in `middle_name_last_name` parsing with rule-based and statistical approaches for Filipino nuances.
2. **Leverage Geographic Data**: Use barangay-level blocking, with fallback to city/province for incomplete data.
3. **Optimize GPU Pipeline**: Implement two-pass GPU strategy for throughput and accuracy.
4. **Ensure Security**: Enforce end-to-end PII protection, especially for CSV and GPU memory.
5. **Iterate on Blocking**: Refine blocking to minimize candidate pairs while maintaining recall.
6. **Plan for Deep Learning**: Reserve R&D for Transformer-based scoring to future-proof the system.
7. **Maximize Podman Benefits**: Leverage rootless execution and Podman’s Kubernetes compatibility for secure, scalable deployment.

## 5. Future Directions
- **Automated Weight Learning**: Use ML to optimize scoring weights.
- **Expanded Linguistic Resources**: Update Filipino name dictionaries continuously.
- **New GPU Architectures**: Optimize kernels for emerging hardware.
- **Graph-Based Resolution**: Explore GNNs for relational data.
- **Active Learning**: Prioritize uncertain pairs for manual review.
- **Advanced PPRL**: Research homomorphic encryption for privacy-preserving matching.

## 6. Conclusion
This plan delivers a state-of-the-art name matching application for Filipino identity data, supporting table-to-table and CSV-to-table matching. By combining MySQL’s indexing, Rust’s performance, GPU acceleration, and Podman’s secure, daemonless containerization, it achieves high accuracy, throughput, and scalability. The phased approach mitigates risks, while comprehensive KPIs ensure success across operational metrics. Podman’s rootless and Kubernetes-compatible design enhances security and deployment flexibility. Continuous R&D will maintain the system’s cutting-edge status, making it a strategic asset for large-scale identity matching.
