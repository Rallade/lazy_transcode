# Lazy Transcode Usage Flowchart

```mermaid
flowchart TD
    Start([Start: lazy_transcode]) --> Setup{Setup Options}
    
    Setup --> Path[Choose Input Path<br/>--path /path/to/videos]
    Path --> Mode{Choose Mode}
    
    Mode --> QP[QP Mode<br/>--mode qp<br/><i>Default</i>]
    Mode --> VBR[VBR Mode<br/>--mode vbr<br/><i>Advanced</i>]
    
    %% QP Mode Branch
    QP --> QPEncoder{Choose Encoder}
    QPEncoder --> QPHardware[Hardware Encoder<br/><i>Auto-detected</i><br/>hevc_amf/hevc_nvenc/hevc_qsv]
    QPEncoder --> QPSoftware[Software Encoder<br/>--cpu<br/>libx265]
    
    QPHardware --> QPStaging{Network Drive?}
    QPSoftware --> QPStaging
    
    QPStaging --> QPLocal[Local Files<br/><i>Direct processing</i>]
    QPStaging --> QPNetwork[Network Drive<br/>--staging<br/><i>3-file buffer optimization</i>]
    
    QPLocal --> QPOptimization[QP Optimization]
    QPNetwork --> QPStagingFlow[Staging Process:<br/>1. Priority transfer first file<br/>2. Start QP optimization immediately<br/>3. Maintain 3-file buffer async]
    QPStagingFlow --> QPOptimization
    
    QPOptimization --> QPMethod{QP Selection Method}
    QPMethod --> QPFixed[Fixed QP<br/>--use-qp 25<br/><i>Skip optimization</i>]
    QPMethod --> QPRange[QP Range Test<br/>--qp-range 20,25,30<br/><i>Test specific values</i>]
    QPMethod --> QPAdaptive[Adaptive Search<br/><i>Default</i><br/>--min-qp 10 --max-qp 45]
    
    QPFixed --> QPSampling{Sample Strategy}
    QPRange --> QPSampling
    QPAdaptive --> QPSampling
    
    QPSampling --> QPAutoSample[Auto-scaled Samples<br/><i>Default: Duration-based</i><br/>--samples 5]
    QPSampling --> QPFixedSample[Fixed Sample Count<br/>--no-auto-scale-samples<br/>--samples 3]
    
    QPAutoSample --> QPDuration{Sample Duration}
    QPFixedSample --> QPDuration
    
    QPDuration --> QPLongSample[Long Samples<br/>--sample-duration 600<br/><i>Default: 10min/stable</i>]
    QPDuration --> QPShortSample[Short Samples<br/>--sample-duration 120<br/><i>Faster testing</i>]
    QPDuration --> QPQuickStart[Quick Start<br/>--quick-sample-duration 60<br/>--quick-iters 3<br/><i>Fast initial, then long</i>]
    
    QPLongSample --> QPExecution[QP Execution]
    QPShortSample --> QPExecution
    QPQuickStart --> QPExecution
    
    %% VBR Mode Branch  
    VBR --> VBREncoder{Choose Encoder}
    VBREncoder --> VBRHardware[Hardware Encoder<br/><i>AMD/NVIDIA/Intel</i><br/>Single-pass VBR]
    VBREncoder --> VBRSoftware[Software Encoder<br/>--cpu<br/>libx265 2-pass VBR]
    
    VBRHardware --> VBRStaging{Network Drive?}
    VBRSoftware --> VBRStaging
    
    VBRStaging --> VBRLocal[Local Files<br/><i>Direct processing</i>]
    VBRStaging --> VBRNetwork[Network Drive<br/>--staging<br/><i>PowerShell copy optimization</i>]
    
    VBRLocal --> VBRConfig[VBR Configuration]
    VBRNetwork --> VBRStagingFlow[VBR Staging Process:<br/>1. Priority transfer first file<br/>2. Start VBR optimization immediately<br/>3. Background staging of remaining files]
    VBRStagingFlow --> VBRConfig
    
    VBRConfig --> VBRParams[VBR Parameters<br/>--vmaf-target 95.0<br/>--vmaf-tol 1.0<br/>--vbr-clips 2<br/>--vbr-clip-duration 30<br/>--vbr-max-trials 6]
    
    VBRParams --> VBRExecution[VBR Execution:<br/>Coordinate descent optimization<br/>Progressive bounds expansion<br/>Bisection search]
    
    %% Execution Paths
    QPExecution --> Processing{Processing Type}
    VBRExecution --> Processing
    
    Processing --> Sequential[Sequential<br/>--no-parallel<br/><i>One file at a time</i>]
    Processing --> Parallel[Parallel<br/><i>Default</i><br/>Multiple files concurrently]
    
    Sequential --> Execution[Execution Mode]
    Parallel --> Execution
    
    Execution --> DryRun[Dry Run<br/>--dry-run<br/><i>Analysis only</i>]
    Execution --> ActualTranscode[Actual Transcoding]
    
    DryRun --> Results[Show Results Only]
    
    ActualTranscode --> OutputLocation{Output Location}
    OutputLocation --> Replace[Replace Originals<br/><i>Default</i><br/>With safety checks]
    OutputLocation --> NonDestructive[Transcoded Subfolder<br/>--non-destructive<br/><i>Keep originals</i>]
    
    Replace --> SafetyChecks[Safety Validation:<br/>VMAF ≥ target<br/>Size ≤ max-size-pct<br/>Quality within bounds]
    NonDestructive --> DirectOutput[Direct Transcoding]
    
    SafetyChecks --> SafetyPass[✓ Passed]
    SafetyChecks --> SafetyFail[✗ Failed]
    
    SafetyFail --> SafetyPrompt{Within Prompt Band?}
    SafetyPrompt --> SafetyInteractive[Interactive Prompt<br/><i>Quality close to threshold</i>]
    SafetyPrompt --> SafetyAbort[Auto Abort<br/><i>Quality too low</i>]
    
    SafetyInteractive --> SafetyManual{User Choice}
    SafetyManual --> SafetyOverride[Override<br/>--force-overwrite]
    SafetyManual --> SafetyCancel[Cancel Processing]
    
    SafetyAbort --> ForceCheck{Force Override?}
    ForceCheck --> SafetyOverride
    ForceCheck --> SafetyCancel
    
    SafetyPass --> DirectOutput
    SafetyOverride --> DirectOutput
    DirectOutput --> Results
    SafetyCancel --> Results
    
    Results --> Complete([Complete])
    
    %% Styling
    classDef modeBox fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    classDef hardwareBox fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px  
    classDef softwareBox fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef stagingBox fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef executionBox fill:#fff8e1,stroke:#ffa000,stroke-width:2px
    classDef safetyBox fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    
    class QP,VBR modeBox
    class QPHardware,VBRHardware hardwareBox
    class QPSoftware,VBRSoftware softwareBox  
    class QPNetwork,VBRNetwork,QPStagingFlow,VBRStagingFlow stagingBox
    class DryRun,ActualTranscode,Sequential,Parallel executionBox
    class SafetyChecks,SafetyFail,SafetyAbort,SafetyOverride safetyBox
```

## Key Decision Points

### 1. **Mode Selection**
- **QP Mode** (`--mode qp`): Traditional quality-based optimization
  - Faster optimization
  - Fixed QP per file 
  - Good for consistent quality

- **VBR Mode** (`--mode vbr`): Advanced bitrate optimization  
  - Coordinate descent algorithm
  - Dynamic bitrate adjustment
  - Better size control

### 2. **Encoder Detection & Override**
- **Auto-detection priority**: AMD → NVIDIA → Intel → Software
- **Force software**: `--cpu` (uses libx265 with advanced threading)
- **Hardware encoders**: 
  - `hevc_amf` (AMD)
  - `hevc_nvenc` (NVIDIA) 
  - `hevc_qsv` (Intel)

### 3. **Network Drive Optimization**
- **Local files**: Direct processing
- **Network drives**: `--staging` enables:
  - PowerShell copy optimization (500 Mbps)
  - 3-file async buffer
  - First-file priority processing
  - Local temp directory staging

### 4. **Quality & Performance Tuning**
- **VMAF targets**: `--vmaf-target 95.0` `--vmaf-min 90.0`
- **Sample optimization**: Auto-scaling or fixed count
- **Threading**: Auto-optimized for CPU utilization
- **Quick start**: Fast initial iterations, then precise

### 5. **Safety & Output Options**
- **Dry run**: `--dry-run` for analysis only
- **Non-destructive**: `--non-destructive` keeps originals
- **Force override**: `--force-overwrite` bypasses safety checks
- **Size limits**: `--max-size-pct 100` prevents size increases

## Common Usage Patterns

### Basic Usage
```bash
python transcode.py --path /path/to/videos
```

### Network Drive Optimization  
```bash
python transcode.py --path "M:\Shows" --staging --mode vbr
```

### High-Quality Analysis
```bash  
python transcode.py --mode vbr --vmaf-target 98 --dry-run
```

### Fast Testing
```bash
python transcode.py --samples 2 --sample-duration 120 --dry-run
```

### Safe Non-Destructive
```bash
python transcode.py --non-destructive --max-size-pct 80
```
