# Documentation of Image Similarity Experiment Unit Test

## Introduction

This document provides a structured guide for understanding, modifying and executing unit tests for the SpatialPoolerImageSimilarityExperiments class. The unit tests validate the Spatial Pooling process in an HTM (Hierarchical Temporal Memory) model, measuring image similarity by comparing Sparse Distributed Representations (SDRs).

## Project Overview
The project consists of the following key components:

-   ##### UnitTestsProject → Root namespace containing unit tests.

- ##### SpatialPoolerImageSimilarityExperiments.cs → Main test file for running the experiments.

- ##### Output Directory → Stores output files generated from test runs.

- ##### Similarity\TestFiles → Contains input images used in similarity testing.

## Environment Setup for testing

Before running the tests, ensure the following dependencies are installed:

-  .NET SDK 6.0+

-  Microsoft.VisualStudio.TestTools.UnitTesting

-  NeoCortex

-  NeoCortexApi



## Spatial Pooler Image Similarity Experiment

### Overview

The unit test performs a Spatial Pooling experiment to analyze image similarity. It processes images through the HTM Spatial Pooler, calculates Hamming distances, and records active column representations.

The test follows a structured workflow:

- Load input images from the dataset.

- Initialize the Spatial Pooler with specific configurations.

- Process images and compute Sparse Distributed Representations (SDRs).

- Measure Hamming distance between SDRs.

- Store results for similarity comparison.


Find the code here: [SpatialPoolerImiageSimilarityExperiments](https://github.com/mushfiq5900/code_crafters/blob/master/source/UnitTestsProject/Similarity/SpatialPoolerImiageSimilarityExperiments.cs)

### Test Class: SpatialPoolerImageSimilarityExperiments

#### Namespace and Class Declaration 

The unit tests are organized within the UnitTestsProject namespace and categorized under "Experiment."
  
    csharp
    
    namespace UnitTestsProject
    {
    [TestClass]
    [TestCategory("Experiment")]
    public class SpatialPoolerImageSimilarityExperiments
    }


### Key Configuration Parameters

The test environment uses the following key settings:

| Parameter         | Description                                 | Example Value  |
|------------------|-------------------------------------|----------------|
|**imgSize**         | Input image size (width × height)  | `28 × 28`        |
| **colDims**        | Column dimensions for the spatial pooler | `{64, 64}`       |
| **numOfCols**      | Total number of columns in the spatial pooler | `64 × 64 = 4096` |
| **maxBoost**       | Maximum boost factor for column activation | `10.0`           |
| **DutyCyclePeriod** | Frequency of duty cycle updates | `100`            |


#### Training Files Configuration
 
The test loads training images from a predefined folder:
    
    csharp
    
    string trainingFolder = "Similarity\\TestFiles";
    string TestOutputFolder = $"Output-{nameof(ImageSimilarityExperiment)}";

    var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.png");




### HtmConfig Configuration Documentation

####  Initialization Parameters

The following describes the configuration parameters used to initialize the HtmConfig object for an HTM-based spatial pooler.

    csharp
     
    HtmConfig cfg = new HtmConfig(new int[] { imgSize, imgSize }, new int[] { numOfCols })
    {
    CellsPerColumn = 10,
    InputDimensions = new int[] { imgSize, imgSize },
    NumInputs = imgSize * imgSize,
    ColumnDimensions = colDims,
    MaxBoost = maxBoost,
    DutyCyclePeriod = 100,
    MinPctOverlapDutyCycles = minOctOverlapCycles,
    GlobalInhibition = false,
    NumActiveColumnsPerInhArea = 0.02 * numOfCols,
    PotentialRadius = (int)(0.15 * imgSize * imgSize),
    LocalAreaDensity = -1,
    ActivationThreshold = 10,
    MaxSynapsesPerSegment = (int)(0.01 * numOfCols),
    Random = new ThreadSafeRandom(42),
    StimulusThreshold = 10,
    };


- **`imgSize`**: The input image size (width × height).
- **`numOfCols`**: Total number of columns in the spatial pooler.
- **`CellsPerColumn`**: Number of cells per column (default: 10).
- **`InputDimensions`**: Dimensions of the input space, set to `{ imgSize, imgSize }`.
- **`NumInputs`**: Total number of input neurons, calculated as `imgSize × imgSize`.
- **`ColumnDimensions`**: The dimensions of the spatial pooler columns, set to `colDims`.
- **`MaxBoost`**: Maximum boost factor applied to column activation.
- **`DutyCyclePeriod`**: The period over which duty cycles are updated (default: 100).
- **`MinPctOverlapDutyCycles`**: Minimum percentage of overlap duty cycles to maintain activation.
- **`GlobalInhibition`**: If `true`, inhibition is applied globally; otherwise, local inhibition is used.
- **`NumActiveColumnsPerInhArea`**: The number of active columns per inhibition area (2% of `numOfCols`).
- **`PotentialRadius`**: The radius defining the pool of potential connections, set to 15% of the input space.
- **`LocalAreaDensity`**: Determines local area inhibition density (-1 means default settings).
- **`ActivationThreshold`**: Minimum overlap required for a column to become active.
- **`MaxSynapsesPerSegment`**: Maximum synapses per segment (1% of `numOfCols`).
- **`Random`**: A thread-safe random number generator initialized with a seed (`42`).
- **`StimulusThreshold`**: Minimum stimulation required for a column to activate.

This configuration ensures an efficient and adaptable spatial pooler for hierarchical temporal memory (HTM) models. 




## Test Cases

### 1. Image Similarity Experiment

Function: `ImageSimilarityExperiment()`

Purpose:

- Run the Spatial Pooler on input images.

- Store results to analyze image similarity.

####
Test Code:

    csharp
        
    [TestMethod]
    public void ImageSimilarityExperiment()
    {
    string inputPrefix = "input_"; 
    string[] images = { "digit1", "digit2", "digit3", "digit4" };

    foreach (string image in images)
    {
        var inputImage = LoadImage($"{inputPrefix}{image}.jpg");
        SpatialPooler sp = new SpatialPooler(cfg);

        var activeColumns = sp.Compute(inputImage);

        LogHammingDistance(activeColumns, image);
        SaveActiveColumns(activeColumns, image);

        Assert.IsTrue(activeColumns.Count > 0);
    }
    }

### 2. Similarity Experiment with Encoder

 Function:  `SimilarityExperimentWithEncoder()`
 
 Purpose:

- Encode numeric inputs into Sparse Distributed Representations (SDRs).

- Verify stability of the spatial pooler.

Test Code:

    csharp
    
    [TestMethod]
    public void SimilarityExperimentWithEncoder()
    {
    string[] inputs = { "0", "1", "2", "3" }; 
    ScalarEncoder encoder = new ScalarEncoder(0, 10);

    foreach (string input in inputs)
    {
        var encodedInput = encoder.Encode(int.Parse(input));

        SpatialPooler sp = new SpatialPooler(cfg);
        var activeColumns = sp.Compute(encodedInput);

        SaveActiveColumns(activeColumns, input);

        Assert.IsTrue(activeColumns.Count > 0);
    }
    }
  
#### 3. Similarity Calculation

Function: `CalculateSimilarity()`

Purpose:

- Compute correlation between SDRs for different inputs.

- Store similarity results in a CSV file.

        csharp
    
        [TestMethod]
    
         public void CalculateSimilarity()
         {
        Dictionary<string, int[]> sdrs = new Dictionary<string, int[]>();
        Dictionary<string, int[]> inputVectors = new Dictionary<string, int[]>();
    
        string[] inputs = { "digit1", "digit2", "digit3", "digit4" };
    
        foreach (string input in inputs)
        {
            var inputImage = LoadImage($"{input}.jpg");
            SpatialPooler sp = new SpatialPooler(cfg);
            var sdr = sp.Compute(inputImage);
    
            sdrs.Add(input, sdr);
            inputVectors.Add(input, inputImage);
        }
    
        var correlations = CalculateCorrelations(sdrs);
        WriteCorrelationCsv(correlations);
        }


## Debugging & Logging

### Debug Output

Use `Debug.WriteLine()` for runtime debugging:


    csharp
    
    Debug.WriteLine($"Cycle: {cycleNumber}, Active Columns: {string.Join(",", activeColumns)}");

### File Logging

Save active column vectors to a log file:

    csharp
    
    StreamWriter writer = new StreamWriter("activeCol.txt", true);
    writer.WriteLine($"Cycle: {cycleNumber}, Active Columns: {string.Join(",", activeColumns)}");

## Expected Outcomes

### **Expected Outcome of HtmConfig Configuration**

The **HtmConfig** object initializes a spatial pooler with the given parameters, ensuring a dynamic and responsive model suitable for processing input data in HTM-based systems. The expected outcome of the configuration is as follows:

#### 1. **Input Handling**
   - **Input Size**: The model is designed to handle input images of size `28 x 28`, where each pixel is represented as an input neuron. This ensures that the spatial pooler works with a 2D grid of `784` neurons (`imgSize x imgSize`).
   - **Input Dimensions**: The spatial pooler processes data with the defined input dimensions (`imgSize x imgSize`), meaning it works with a 2D grid of neurons, such as images.

#### 2. **Column Setup**
   - **Total Columns**: The spatial pooler is configured with a total of `4096` columns (64 x 64), which allows for rich and detailed spatial representations.
   - **Column Dimensions**: The column dimensions (`colDims`) are set to `{64, 64}`, defining the 2D grid of columns that represents the space in which the HTM model operates.

#### 3. **Cell Setup**
   - **Cells Per Column**: With `10` cells per column, each column can represent multiple spatial patterns (cells), offering a more nuanced learning capability.
   
#### 4. **Boost Mechanism**
   - **Max Boost**: A maximum boost factor of `10.0` is applied to the columns, which allows for adaptive boosting when certain columns are not activated frequently, thus maintaining competitive learning dynamics in the model.
   
#### 5. **Duty Cycle Management**
   - **Duty Cycle Updates**: Duty cycles are updated every `100` time steps, maintaining a balance between stability and responsiveness. This ensures that the model remains adaptive while avoiding overfitting or excessive plasticity.
   - **MinPctOverlapDutyCycles**: The model ensures a minimum overlap in duty cycles, which helps in maintaining effective patterns in the activation state over time.

#### 6. **Inhibition**
   - **Global Inhibition**: Set to `false`, which means inhibition will occur locally within regions of the spatial pooler. This allows for greater flexibility and specificity in learning and pattern recognition.
   - **Num Active Columns Per Inhibition Area**: With `0.02 * numOfCols`, this value ensures that a small but consistent fraction of columns are active per inhibition area, helping maintain diversity and preventing overfitting to any one pattern.

#### 7. **Connectivity and Synapses**
   - **Potential Radius**: The potential radius of `15%` of the input space ensures that each column can potentially connect to a subset of other columns, maintaining a level of local connectivity while also promoting a global representation.
   - **Max Synapses Per Segment**: The number of synapses per segment is limited to `1% of numOfCols`, ensuring that the connections between columns are sparse and efficient.

#### 8. **Column Activation**
   - **Activation Threshold**: The column becomes active only if it has a minimum of `10` active synapses, enforcing a certain threshold for column activation. This prevents noise from triggering spurious activations.
   
#### 9. **Stimulation and Learning**
   - **Stimulus Threshold**: The stimulus threshold ensures that a column is only activated when the input sufficiently stimulates it. This prevents spurious activation in columns that are not meaningfully engaged with the input data.
   
#### 10. **Randomness and Variability**
   - **Randomization**: The random number generator, initialized with a seed value of `42`, ensures reproducibility in the learning process, while still introducing variability in connections and column activations to promote diversity in learning patterns.



#### **Summary of Expected Outcomes**
By using the above configuration, the spatial pooler is expected to:

- Effectively process input data in the form of a 28x28 grid of neurons.
- Use local inhibition mechanisms to enhance the learning process and avoid overfitting.
- Utilize duty cycle updates to ensure the model adapts over time and maintains meaningful activations.
- Ensure efficient column activation through a combination of activation thresholds, synapse limits, and potential radius.
- Exhibit a robust and adaptable learning system with a balance of randomness (for diversity) and stability (through duty cycle management and boosting).

Overall, this configuration provides a finely tuned setup for learning and recognition tasks in a hierarchical temporal memory system.


