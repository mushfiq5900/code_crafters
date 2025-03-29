# ML 24/25-04 Implement the visualization of permanence value




####   In this Documentation we will describe our contribution in this project.
#### Instruction for Running the Project
- Clone the Repository and Run
- You will get the project here
[NeoCortexApiSample](https://github.com/mushfiq5900/code_crafters/tree/master/source/Samples/NeoCortexApiSample)
#### Two Experiments
- **`SpatialPatternLearning.cs`**: Numerical Inputs 
[SpatialPatternLearning.cs](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/SpatialPatternLearning.cs)
- **`ImageBinarizerSpatialPattern.cs`**: Image Inputs 
[ImageBinarizerSpatialPattern.cs](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/ImageBinarizerSpatialPattern.cs)

#### Image Sample Folder: [Check Here](https://github.com/mushfiq5900/code_crafters/tree/master/source/Documents_Code_Crafters/Sample/TestFiles) 



#### Usages

This folder contains six sample images that are required for running the `ImageBinarizerSpatialPattern.cs` program. The images are necessary for testing and demonstrating the functionality of the program.
After cloning the repository, you need to manually copy and paste this folder into the following location (source\Samples\NeoCortexApiSample\bin\Debug\net9.0)  before executing the `ImageBinarizerSpatialPattern.cs program`: 

#### Notes

- Ensure that all six images remain intact and unmodified.
- The program may not function correctly if the images are missing or stored in a different directory.
- If you encounter any issues, verify that the folder is correctly placed in the specified directory.
 
###### Simply Change the Running commands here 
- **`Program.cs`**: Goto Program.cs file of NeoCortexApiSample
- Change the codes here Clieck the Link below and it will Redirect you.
[Program.cs](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/Program.cs#L23-L28)
###### All the output will be saved here
- code_crafters/tree/master/source/Samples/NeoCortexApiSample\bin\Debug\net9.0


# Introduction

Hierarchical Temporal Memory (HTM) is a machine learning model inspired by the structure and function of the neocortex. It aims to replicate the brain's ability to recognize patterns, make predictions, and learn from sensory data using sparse distributed representations (SDRs). The Spatial Pooler, a key component of HTM, is responsible for creating these SDRs by encoding input patterns into sparse, high-dimensional representations. This process allows the model to capture important features of the input while maintaining efficiency and robustness.

However, while HTM provides a powerful framework for understanding and simulating brain-like learning processes, its inner workings—specifically, the dynamics of synaptic permanence (the strength of connections between neurons)—remain difficult to interpret. Understanding how these permanence values evolve over time is crucial for gaining insights into the learning process and improving the model’s performance.

This paper presents an enhanced visualization approach to track and analyze the evolution of permanence values within the Spatial Pooler. By utilizing the Neocortex API’s Reconstruct method, we are able to monitor changes in permanence and visualize how synaptic stability and learning dynamics unfold. This approach improves the interpretability of HTM networks by providing a clearer understanding of how the model learns and adapts over time.

In addition to examining permanence dynamics, we extend our method by incorporating image data as input to the Spatial Pooler, allowing us to explore the model's potential for visual pattern recognition and reconstruction tasks. Through detailed analysis, we investigate how different encoding strategies and noise levels affect the permanence evolution and overall performance of the model.

Our work contributes to the growing field of HTM research by offering a more transparent and interpretable visualization of the learning mechanisms, facilitating better optimization and understanding of HTM-based systems.

# Methodology 

Our approach focuses on accurately reconstructing original input data using the Hierarchical Temporal Memory (HTM) framework, specifically utilizing the Neocortex API’s Reconstruct() method. We process various input data types, including numerical values (0–99) and images, by first converting them into binary representations using an Image Binarizer. These binary-encoded inputs are then transformed into integer arrays (int[]), which serve as inputs to the HTM Spatial Pooler to generate Sparse Distributed Representations (SDRs).

-   Reconstruction Process: Once the SDRs are generated, the reconstruction begins by using the Reconstruct() method. This method aims to reverse the encoding process and restore the original input from the SDRs based on permanence values. The accuracy of the reconstruction is evaluated using heatmaps and similarity comparisons between the original and reconstructed arrays, along with a Jaccard similarity measure.

- Data Types and Encoding: 

   - Numerical Data: Each numerical value (0-99) is encoded into a 200-bit representation, which is then transformed into an int[] array for input into the HTM Spatial Pooler.

   - Image Data: Image data is binarized and converted into binary arrays based on pixel values (e.g., a 28x28 image becomes a 784-element array). These binary arrays are processed similarly to the numerical data.

- Workflow of the Reconstruction Method:

  - Validation: The method begins by validating the input to ensure no null values are present.

   - Column Retrieval: It retrieves the active columns associated with the input data and computes the permanence values.

   - Reconstruction: The process reconstructs the input data by accumulating permanence values from active mini-columns, mapping them back to the original structure.

- Visualization and Analysis : We visualize the reconstructed permanence values using heatmaps and similarity plots. These visualizations help assess how accurately the HTM framework captures the input data’s structure, both for numerical and image inputs. The similarity between the original and reconstructed data is quantified using Jaccard similarity, providing a clear measure of reconstruction fidelity.

This approach helps understand how HTM learns from numerical and image data, revealing how effectively it preserves the input structure during reconstruction.


![Methodology Flowchart](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/Representation%20of%20the%20Experiment%402x-8.png)

*Figure 1: Methodology Flowchart*                   




## Reconstruct() Method:

In the HTM framework, the Reconstruct method reverses the transformation of input data into Sparse Distributed Representations (SDRs) by the Spatial Pooler, approximating the original input from activated SDRs and providing insight into how information is encoded and preserved.

    csharp

    public Dictionary<int, double> Reconstruct(int[] activeMiniColumns)
    {
     if (activeMiniColumns == null)
     {
         throw new ArgumentNullException(nameof(activeMiniColumns));
     }

     var cols = connections.GetColumnList(activeMiniColumns);

     Dictionary<int, double> permancences = new Dictionary<int, double>();

    
     foreach (var col in cols)
     {
         col.ProximalDendrite.Synapses.ForEach(s =>
         {
             double currPerm = 0.0;

             
             if (permancences.TryGetValue(s.InputIndex, out currPerm))
             {
               
                 permancences[s.InputIndex] = s.Permanence + currPerm;
             }
             else
             {
              
                 permancences[s.InputIndex] = s.Permanence;
             }
         });
     }

     return permancences;
    }

[Reconstruction in SP](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexApi/SpatialPooler.cs#L1442-L1482) - Lines (1442 to 1482)

## Reconstruct() Workflow:
- **Input Validation:** The method starts by validating the input array, ensuring it is not null before proceeding with the reconstruction.
   
- **Column Retrieval:** It identifies the active mini-columns by using the `activeMiniColumns` array, and retrieves the corresponding permanence values from the network's connections.

- **Key Components:** Proximal dendrites, which receive input and identify spatial patterns, and synapses, which connect dendrites to other neurons and carry permanence values that define connection strength.
   
- **Reconstruction Process:** The method iterates through active mini-columns, collecting permanence values for each synapse into a dictionary. The dictionary is returned, representing the reconstructed input.

This method plays a vital role in restoring the original input data from the network’s learned representations, aiding in the analysis of the HTM model’s internal workings.

## Running Reconstruct Method for Numerical Inputs

    csharp
    
    private void RunRustructuringExperiment(SpatialPooler sp, EncoderBase encoder,List<double> inputValues)
    {
     List<List<double>> heatmapData = new List<List<double>>();
     List<int[]> normalizedPermanence = new List<int[]>();
     List<int[]> encodedInputs = new List<int[]>();
     List<double[]> similarityList = new List<double[]>();

     string directoryPath = Path.Combine(Directory.GetCurrentDirectory(), "Heatmaps");
     if (!Directory.Exists(directoryPath))
     {
         Directory.CreateDirectory(directoryPath);
     }

     foreach (var input in inputValues)
     {
         var inpSdr = encoder.Encode(input);
         var actCols = sp.Compute(inpSdr, false);
         Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(actCols);

         int maxInput = inpSdr.Length;
         Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
         foreach (var kvp in reconstructedPermanence)
         {
             allPermanenceDictionary[kvp.Key] = kvp.Value;
         }

         for (int inputIndex = 0; inputIndex < maxInput; inputIndex++)
         {
             if (!allPermanenceDictionary.ContainsKey(inputIndex))
             {
                 allPermanenceDictionary[inputIndex] = 0.0;
             }
         }

         var sortedAllPermanenceDictionary = allPermanenceDictionary.OrderBy(kvp => kvp.Key);
         List<double> permanenceValuesList = sortedAllPermanenceDictionary.Select(kvp => kvp.Value).ToList();
         heatmapData.Add(permanenceValuesList);

         Debug.WriteLine($"Input: {input} SDR: {Helpers.StringifyVector(actCols)}");

         var ThresholdValue = 8.3;
         List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);
         normalizedPermanence.Add(normalizePermanenceList.ToArray());

         encodedInputs.Add(inpSdr);

         var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inpSdr, normalizePermanenceList.ToArray());
         double[] similarityArray = new double[] { similarity };
         similarityList.Add(similarityArray);
     }

     GenerateMatrics(heatmapData);
     GenerateEncodedMatrics(encodedInputs.Select(arr => arr.ToList()).ToList());
     GenerateReconstructedMatrics(normalizedPermanence.Select(arr => arr.ToList()).ToList());
     DrawSimilarityPlots(similarityList);

     Console.WriteLine("All heatmaps generated and similarity plots saved.");
    }


[Running Reconstruct Method For Numeric Data](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/SpatialPatternLearning.cs#L156-L213) - Lines (156 to 213)
## Running Reconstruct Method for Image Inputs

    csharp
    
    private void RunRustructuringExperiment(SpatialPooler sp, List<int[]> activeColsList, string[] trainingImages)
    {
    List<int[]> normalizedPermanence = new List<int[]>();
    Dictionary<string, double> highestSimilarityPerImage = new Dictionary<string, double>(); // Stores highest similarity for each image

    int totalCycles = 1; // Number of cycles

    for (int cycleIndex = 0; cycleIndex < totalCycles; cycleIndex++)
    {
        if (cycleIndex == totalCycles - 1) // Process only the last cycle
        {
            foreach (var actcols in activeColsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct permanence for SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(actcols);
                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();

                foreach (var kvp in reconstructedPermanence)
                {
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52;

                // Assign inactive columns permanence = 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values
                var ThresholdValue = 69.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                int[] currentNormalizedPermanence = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue).ToArray();

                normalizedPermanence.Add(currentNormalizedPermanence);
                // Define a unique image index for consistency
                int imageIndex = activeColsList.IndexOf(actcols);

                // Generate consistent names for both images
                string reconstructedImageName = $"ReconstructedImage_{imageIndex}";
                string heatmapImageName = $"Heatmap_{imageIndex}";

                // Save the reconstructed binary image
                NeoCortexUtils.GenarateReconstrucetedBinarizedImage(currentNormalizedPermanence.ToArray(), reconstructedImageName);

                // *Save heatmap per cycle*
                List<List<double>> heatmapData = new List<List<double>> { permanenceValuesList }; // Use sorted permanence
                string folderPath = Path.Combine(Directory.GetCurrentDirectory(), "ImageInputHeatmaps");
                Directory.CreateDirectory(folderPath);
                string heatmapFilePath = Path.Combine(folderPath, $"{heatmapImageName}.png");

                NeoCortexUtils.GenarateImageInputHeatmap(heatmapData, heatmapFilePath);
                GenerateMatricsforImage(heatmapData);

                // Compare similarity for each training image
                for (int i = 0; i < trainingImages.Length; i++)
                {
                    string imageName = Path.GetFileNameWithoutExtension(trainingImages[i]);
                    string binarizedFile = Path.Combine("Binarized", $"{imageName}.txt");

                    if (File.Exists(binarizedFile))
                    {
                        // Read the binarized image (input vector)
                        int[] inputVector = NeoCortexUtils.ReadCsvIntegers(binarizedFile).ToArray();

                        // Compute Jaccard Similarity
                        double jaccardSim = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, currentNormalizedPermanence);

                        // Track the highest similarity for this image
                        if (!highestSimilarityPerImage.ContainsKey(imageName) || jaccardSim > highestSimilarityPerImage[imageName])
                        {
                            highestSimilarityPerImage[imageName] = jaccardSim;
                        }

                        Debug.WriteLine($"Image {imageName}.jpg ========= Jaccard Similarity: {jaccardSim}");
                    }
                    else
                    {
                        Debug.WriteLine($"Warning: Binarized file {binarizedFile} not found.");
                    }
                }
            }
        }
    }

    // Pass only the highest similarity values for the training images
    List<double> finalSimilarityValues = highestSimilarityPerImage.Values.ToList();
    DrawSimilarityPlots(finalSimilarityValues);
    }


[Running Reconstruct Method for Image Data](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/ImageBinarizerSpatialPattern.cs#L156-L250) - Lines (156 to 250)
## Implementation Details for Numeric and Image Input Types:

The RunRustructuringExperiment method is designed to process both numeric and image input types using the Hierarchical Temporal Memory (HTM) framework, particularly the Spatial Pooler and Reconstruct method for data processing and analysis. The following sections detail how the experiment is implemented for numeric and image data inputs.

### 1. Numeric Input Processing 
For numeric data, the experiment performs the following steps: 
     
      
     
     
#### 1.1 Input Encoding and Spatial Pooling  

     
     
     csharp
        
     var inpSdr = encoder.Encode(input);
     var actCols = sp.Compute(inpSdr, false);
          
- Input Encoding: Each numeric input is encoded into a Sparse Distributed Representation (SDR) using the `encoder.Encode(input)` method.

- Spatial Pooling: The encoded SDR is processed by the Spatial Pooler using `sp.Compute(inpSdr, false)` , producing the active columns.   

    
#### 1.2 Reconstruction of Permanence Values
   
  
   
     csharp
   
     Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(actCols);
   

  -  Reconstruction: The Reconstruct method is invoked on the active columns to approximate the permanence values, representing the connections between columns and input.

#### 1.3 Normalization of Permanence Values
 
   
     csharp
     
     var ThresholdValue = 8.3;
     List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);
   
   - Normalization: A threshold value (e.g., ThresholdValue = 8.3) is applied to the permanence values using ThresholdingProbabilities, which normalizes the values by setting less significant values to zero.

#### 1.4 Jaccard Similarity Comparison

 
    csharp
     
    var similarity = MathHelpers.JaccardSimilarityofBinaryArrays(inpSdr, normalizePermanenceList.ToArray());
   

-  Jaccard Similarity: The Jaccard Similarity metric is used to compare the original SDRs and the normalized permanence values, providing a measure of their similarity.


#### 1.5 Data Visualization and Metrics Generation

 
    csharp
     
    GenerateMatrics(heatmapData);
    GenerateEncodedMatrics(encodedInputs.Select(arr => arr.ToList()).ToList());
    GenerateReconstructedMatrics(normalizedPermanence.Select(arr => arr.ToList()).ToList());
    DrawSimilarityPlots(similarityList);

   
- Visualization: Heatmaps and similarity plots are generated from the reconstructed permanence values, SDRs, and other processed data to help visualize the accuracy of the reconstruction.


### 2. Image Input Processing

For image data, the method follows a similar process with a few additional steps specific to image handling:

#### 2.1 Binarization and Spatial Pooling


  
    csharp
     
    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(binarizedFile).ToArray();
    var actCols = sp.Compute(inputVector, false);

  
- Binarization: Image data is binarized before being encoded. The binarized images are processed through the Spatial Pooler to compute the active columns, similar to numeric data processing.

#### 2.2 Reconstruction of Permanence Values

    
    csharp
     
    Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(actcols);

   
- Reconstruction: The active columns corresponding to the binarized image are used to reconstruct the permanence values.    
    
#### 2.3 Normalization of Permanence Values
   
    csharp

    var ThresholdValue = 69.0;
    List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
    int[] currentNormalizedPermanence = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue).ToArray();


- Normalization: A higher threshold (`ThresholdValue = 69.0`) is applied to the permanence values to normalize the data. This step ensures that only the most significant permanence values remain active.

#### 2.4 Image Saving and Heatmap Generation
   
    csharp

    NeoCortexUtils.GenarateReconstrucetedBinarizedImage(currentNormalizedPermanence.ToArray(), reconstructedImageName);
    NeoCortexUtils.GenarateImageInputHeatmap(heatmapData, heatmapFilePath);

- Reconstructed Image Saving: The reconstructed binary image is saved using `GenarateReconstrucetedBinarizedImage` .

- Heatmap Saving: A heatmap of the image is generated and saved using `GenarateImageInputHeatmap` .

#### 2.5 Jaccard Similarity Comparison for Images


    csharp

    double jaccardSim = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, currentNormalizedPermanence);

- Jaccard Similarity: Similar to numeric data, the Jaccard Similarity is computed for each image by comparing the reconstructed permanence values to the binarized images in the training set.

#### 2.6 Tracking the Highest Similarity per Image
 
    csharp
 
    if (!highestSimilarityPerImage.ContainsKey(imageName) || jaccardSim > highestSimilarityPerImage[imageName])
    {
    highestSimilarityPerImage[imageName] = jaccardSim;
    }

- Tracking Highest Similarity: For each image, the highest similarity value with the reconstructed permanence values is stored and updated as necessary.


## Generates heatmap matrics from the collected experiment data
    csharp
    
    private void GenerateMatrics(List<List<double>> heatmapData)
    {
    int i = 1;
    foreach (var values in heatmapData)
    {
        string folderPath = Path.Combine(Environment.CurrentDirectory, "HeatMapMatrics");
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }

        string filePath = Path.Combine(folderPath, $"heatmapMatrix_{i}.png");

        Debug.WriteLine($"FilePath: {filePath}");

        int rows = 8;
        int cols = 25;

        if (values.Count != rows * cols)
        {
            Debug.WriteLine("Data does not match expected size of 8x25.");
            continue;
        }

        NeoCortexUtils.SaveHeatmapValuesAsImage(values, filePath, rows, cols, 50);

        Debug.WriteLine($"Heatmap {i} generated and saved successfully.");
        i++;
    }
    }

[GenarateMatrics Function](https://github.com/mushfiq5900/code_crafters/blob/master/source/Samples/NeoCortexApiSample/SpatialPatternLearning.cs#L218-L247) - Lines (218 to 247)


The GenerateMetrics method generates and saves heatmap matrices as images based on provided numerical data. It ensures proper directory creation and organizes heatmaps for visualization.

###### Parameters
- `heatmapData` (List<List<double>>): A list of numerical values representing the heatmap data for different inputs.

###### Functionality

1. Iterates through the provided heatmapData.

2. Creates a folder (HeatMapMatrics) to store heatmap images if it does not exist.

3. Defines a fixed grid size (8x25) for the heatmap representation.

4. Validates that the data matches the expected size; otherwise, logs a warning.

5. Calls NeoCortexUtils.SaveHeatmapValuesAsImage to generate and save the heatmap image.

6. Saves the heatmap as a .png file with an incremental naming pattern (heatmapMatrix_1.png, heatmapMatrix_2.png, etc.).

7. Logs the success or failure of each heatmap generation.

##### Usage
This method is useful for visualizing the permanence values reconstructed during HTM processing, aiding in debugging and analysis.


## Draws bitmaps from list of arrays Function

    csharp
    
    public static void DrawBitmaps(List<int[,]> twoDimArrays, String filePath, Color inactiveCellColor, Color activeCellColor, int bmpWidth = 1024, int bmpHeight = 1024)
    {
    int widthOfAll = 0, heightOfAll = 0;

    foreach (var arr in twoDimArrays)
    {
        widthOfAll += arr.GetLength(0);
        heightOfAll += arr.GetLength(1);
    }

    if (widthOfAll > bmpWidth || heightOfAll > bmpHeight)
        throw new ArgumentException("Size of all included arrays must be less than specified 'bmpWidth' and 'bmpHeight'");

    System.Drawing.Bitmap myBitmap = new System.Drawing.Bitmap(bmpWidth, bmpHeight);
    int k = 0;

    for (int n = 0; n < twoDimArrays.Count; n++)
    {
        var arr = twoDimArrays[n];

        int w = arr.GetLength(0);
        int h = arr.GetLength(1);

        var scale = ((bmpWidth) / twoDimArrays.Count) / (w + 1);// +1 is for offset between pictures in X dim.

        //if (scale * (w + 1) < (bmpWidth))
        //    scale++;

        for (int Xcount = 0; Xcount < w; Xcount++)
        {
            for (int Ycount = 0; Ycount < h; Ycount++)
            {
                for (int padX = 0; padX < scale; padX++)
                {
                    for (int padY = 0; padY < scale; padY++)
                    {
                        if (arr[Xcount, Ycount] == 1)
                        {
                            myBitmap.SetPixel(n * (bmpWidth / twoDimArrays.Count) + Xcount * scale + padX, Ycount * scale + padY, activeCellColor); // HERE IS YOUR LOGIC
                            k++;
                        }
                        else
                        {
                            myBitmap.SetPixel(n * (bmpWidth / twoDimArrays.Count) + Xcount * scale + padX, Ycount * scale + padY, inactiveCellColor); // HERE IS YOUR LOGIC
                            k++;
                        }
                    }
                }
            }
        }
    }

    myBitmap.Save(filePath, ImageFormat.Png);
    }

[DrawBitmaps Function](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L154-L207) - Lines (157 to 207)

### Parameters Documentation for `DrawBitmaps` Method

- `twoDimArrays` (List<int[,]>): A collection of 2D binary arrays where each array represents an image. The method converts these into a visual bitmap representation.

- `filePath` (string): The destination path for saving the generated bitmap image in PNG format.

- `inactiveCellColor` (Color): Defines the color for inactive cells (0s) in the bitmap.

- `activeCellColor` (Color): Defines the color for active cells (1s) in the bitmap.

- `bmpWidth` (int, optional, default = 1024): Specifies the width of the output bitmap image. Must be large enough to accommodate all input arrays.

- `bmpHeight` (int, optional, default = 1024): Specifies the height of the output bitmap image. Must be large enough to fit all included arrays.

This method scales and arranges multiple binary matrices into a single image while maintaining proper spacing and proportions. It throws an exception if the total array size exceeds the defined image dimensions.


# Genarate DrawHeatmap Function

    csharp
    
    public static void DrawHeatmaps(List<double[,]> twoDimArrays, String filePath,
    int bmpWidth = 1024,
    int bmpHeight = 1024,
    decimal redStart = 200, decimal yellowMiddle = 127, decimal greenStart = 20)
    {
    int widthOfAll = 0, heightOfAll = 0;

    foreach (var arr in twoDimArrays)
    {
        widthOfAll += arr.GetLength(0);
        heightOfAll += arr.GetLength(1);
    }

    if (widthOfAll > bmpWidth || heightOfAll > bmpHeight)
        throw new ArgumentException("Size of all included arrays must be less than specified 'bmpWidth' and 'bmpHeight'");

    System.Drawing.Bitmap myBitmap = new System.Drawing.Bitmap(bmpWidth, bmpHeight);
    int k = 0;

    for (int n = 0; n < twoDimArrays.Count; n++)
    {
        var arr = twoDimArrays[n];

        int w = arr.GetLength(0);
        int h = arr.GetLength(1);

        var scale = Math.Max(1, ((bmpWidth) / twoDimArrays.Count) / (w + 1));// +1 is for offset between pictures in X dim.

        for (int Xcount = 0; Xcount < w; Xcount++)
        {
            for (int Ycount = 0; Ycount < h; Ycount++)
            {
                for (int padX = 0; padX < scale; padX++)
                {
                    for (int padY = 0; padY < scale; padY++)
                    {
                        myBitmap.SetPixel(n * (bmpWidth / twoDimArrays.Count) + Xcount * scale + padX, Ycount * scale + padY, GetColor(redStart, yellowMiddle, greenStart, (Decimal)arr[Xcount, Ycount]));
                        k++;
                    }
                }
            }
        }
    }

    myBitmap.Save(filePath, ImageFormat.Png);
    }

[DrawHeatmaps Function](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L415-L460) - Lines (415 to 460)

- Description: The `DrawHeatmaps` method generates and saves a heatmap visualization from a collection of 2D numerical arrays. Each array represents a data matrix where numerical values are mapped to colors using a red-yellow-green gradient, making it easier to interpret variations in the data. The function ensures the proper scaling of multiple heatmaps within the specified image dimensions and outputs the visualization as a PNG file.

- Parameters:
  
  - `twoDimArrays (List<double[,]>)`: A list of 2D numerical arrays, where each array represents a heatmap.

  - `filePath (String)`: The path where the generated heatmap image will be saved.

  - `bmpWidth (int, optional)`: Width of the output heatmap image (default: 1024 px).

  - `bmpHeight (int, optional)`: Height of the output heatmap image (default: 1024 px).

  - `redStart (decimal, optional)`: Red intensity value for high data points (default: 200).

  - `yellowMiddle (decimal, optional)`: Yellow intensity value for mid-range data points (default: 127).

  - `greenStart (decimal, optional)`: Green intensity value for low data points (default: 20).

- Functionality:
  
    - Iterates through all provided 2D arrays to determine the dimensions of the final heatmap.

    - Validates the total size to ensure it does not exceed the specified image dimensions `(bmpWidth, bmpHeight)`.

    - Maps numerical values to colors using the defined red-yellow-green gradient for clear visual representation.

    - Scales the heatmap appropriately to maintain consistent proportions for each dataset.

    - Generates a PNG image with the final heatmap and saves it to the specified file path.
    
## Similarity Calculation Using Jaccard Similarity Coefficient
    csharp
    
    public static double JaccardSimilarityofBinaryArrays(int[] arr1, int[] arr2)
    {
    if (arr1.Length != arr2.Length)
    {
        throw new ArgumentException("Arrays must have the same length.");
    }

    int intersectionCount = 0;
    int unionCount = 0;

    for (int i = 0; i < arr1.Length; i++)
    {
        if (arr1[i] == 1 && arr2[i] == 1)
        {
            intersectionCount++;
        }
        if (arr1[i] == 1 || arr2[i] == 1)
        {
            unionCount++;
        }
    }

    return (double)intersectionCount / unionCount;
    }

Here is the Function
[MathHelpers.cs](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexApi/Utility/MathHelpers.cs#L182-L207) - Lines (182 to 207)

## Genarate Similarity Graph

We Applied this Function to draw Similarity Plot For numerical Input
Click Below for More Details 
[SimilarityPlotForNumericInput](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L678-L786) 

**Outcomes:**


![Final Outcome](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/similarity_plot_for_numerical_value.png)

*Figure 2: Bar graphs of similarity for numerical inputs*


We Applied this Function to draw Similarity Plot For Image Input
Click Below for More Details 
[SimilarityPlotForImageInput](https://github.com/mushfiq5900/code_crafters/blob/master/source/NeoCortexUtils/NeoCortexUtils.cs#L678-L786) 

**Outcomes:**


![Final Outcome](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/similarity_plot_for_image.png)

*Figure 3: Bar graphs of similarity for Image inputs*

# Final Results:

### Results for Numerical Inputs:

![Output heatmap matrics](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/Output%20matrics%20for%20heatmap.png)

*Figure 4: Output heatmap matrics*

![Output heatmap for numerical input](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/Output%20heatmap%20for%20numerical%20input.png)

*Figure 5: Output heatmap for numerical input*

### Results for Image Inputs:
![Reconstructed matrics for image input](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/Reconstructed%20matrics%20for%20image%20input.png)

*Figure 6: Reconstructed matrics for image input*


![Output heatmap for image input](https://raw.githubusercontent.com/mushfiq5900/code_crafters/refs/heads/master/source/Documents_Code_Crafters/Assets/Output%20heatmap%20for%20image%20input.png)

*Figure 7: Output heatmap for image input*


# Conclusion:

This research introduced an enhanced visualization approach for analyzing permanence value dynamics in the HTM Spatial Pooler. Using the Neocortex API’s Reconstruct() method, we effectively represented permanence evolution through heatmaps and similarity graphs for numerical and image inputs. These visualizations provided valuable insights into HTM’s learning behavior, synaptic stability, and reconstruction accuracy, laying the groundwork for future optimizations and applications of HTM networks.

















