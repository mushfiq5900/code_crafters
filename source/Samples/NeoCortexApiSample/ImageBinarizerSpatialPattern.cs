using NeoCortex;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using NeoCortexApi;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeoCortexApiSample
{
    internal class ImageBinarizerSpatialPattern
    {
        // The prefix for input image filenames to process
        public string inputPrefix { get; private set; }

        /// <summary>
        /// The entry point for the Image Binarizer Spatial Pattern Experiment.
        /// This method runs the experiment which learns spatial patterns and demonstrates how the Spatial Pooler (SP) learns from binarized images.
        /// </summary>
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageBinarizerSpatialPattern)}");

            // Experiment parameters
            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 84 * 84;
            int imageSize = 52;
            var colDims = new int[] { 84, 84 };

            // Configuration for HTM (Hierarchical Temporal Memory)
            HtmConfig cfg = new HtmConfig(new int[] { imageSize, imageSize }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                InputDimensions = new int[] { imageSize, imageSize },
                NumInputs = imageSize * imageSize,
                ColumnDimensions = colDims,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = false,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * imageSize * imageSize),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            string trainingFolder = "Sample\\TestFiles";  // Folder with images
            string binarizedFolder = "Binarized";         // Folder to save binarized images
            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.jpg");
            Debug.WriteLine($"[INFO] Found {trainingImages.Length} training images.");

            // Run Experiment and get active columns per image
            var (sp, activeColumnsPerImage) = RunExperiment(cfg, inputPrefix);

            // Run Reconstruction using precomputed active columns
            RunRustructuringExperiment(sp, activeColumnsPerImage, trainingImages);
        }

        /// <summary>
        /// Runs the main experiment to process the training images and train the Spatial Pooler (SP).
        /// It iteratively learns spatial patterns from the binarized images and stores the active columns.
        /// </summary>
        /// <param name="cfg">The configuration for the HTM model.</param>
        /// <param name="inputPrefix">The prefix for input image filenames to process.</param>
        /// <returns>A tuple containing the trained Spatial Pooler and a list of active columns for each image.</returns>
        private (SpatialPooler, List<int[]>) RunExperiment(HtmConfig cfg, string inputPrefix)
        {
            Debug.WriteLine("[INFO] Initializing Experiment...");
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numColumns = 84 * 84; // SDR size

            string trainingFolder = "Sample\\TestFiles";  // Folder with images
            string binarizedFolder = "Binarized";         // Folder to save binarized images
            Directory.CreateDirectory(binarizedFolder);

            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.jpg");
            Debug.WriteLine($"[INFO] Processing {trainingImages.Length} images...");

            int imgSize = 52;  // Resized image size
            string testName = "test_image";

            // Initialize Homeostatic Plasticity and SP
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, trainingImages.Length * 50,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    isInStableState = isStable;
                    Debug.WriteLine($"STABLE: {isStable}, Patterns: {numPatterns}, Inputs: {seenInputs}");
                }, requiredSimilarityThreshold: 0.975
            );

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            List<int[]> activeColsList = new List<int[]>(); // Store active columns for all images
            int[] activeArray = new int[numColumns];
            int maxCycles = 10;
            int currentCycle = 0;

            // Loop to train the model until stable state is reached
            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"Processing Training Cycle for each Image {currentCycle}");

                foreach (var image in trainingImages)
                {
                    // 1. Binarize Image and Save
                    string binarizedFile = Path.Combine(binarizedFolder, $"{Path.GetFileNameWithoutExtension(image)}.txt");
                    if (!File.Exists(binarizedFile)) // Avoid reprocessing
                    {
                        string generatedFile = NeoCortexUtils.BinarizeImage(image, imgSize, testName);
                        File.Copy(generatedFile, binarizedFile, true);
                    }

                    // 2. Read Binarized Image as Input Vector
                    int[] inputVector = NeoCortexUtils.ReadCsvIntegers(binarizedFile).ToArray();

                    for (int cycle = 0; cycle < maxCycles; cycle++)
                    {
                        Array.Clear(activeArray, 0, activeArray.Length);
                        sp.compute(inputVector, activeArray, true);
                        var activeCols = ArrayUtils.IndexWhere(activeArray, el => el == 1);

                        activeColsList.Add(activeCols);  // Store active columns

                        Debug.WriteLine($"[INFO] Cycle {currentCycle}: Processing Image {Path.GetFileName(image)}");
                        Debug.WriteLine($"[DETAILS] Input Vector Length: {inputVector.Length}, Active Columns: {activeCols.Length}");
                        Debug.WriteLine($"[TRACE] Input Vector Sample: {string.Join(",", inputVector.Take(30))} ...");
                        Debug.WriteLine($"[TRACE] Active Columns Sample: {string.Join(",", activeCols.Take(30))} ...");
                        Debug.WriteLine("==============================================");
                    }
                }

                currentCycle++;
            }

            // Pass Active Columns to Reconstruction Experiment
            RunRustructuringExperiment(sp, activeColsList, trainingImages);

            return (sp, activeColsList);
        }

        /// <summary>
        /// Runs the reconstruction experiment using the trained Spatial Pooler (SP) and the active columns learned during the experiment.
        /// This method reconstructs the permanence values for each image and generates heatmaps and similarity plots.
        /// </summary>
        /// <param name="sp">The trained Spatial Pooler.</param>
        /// <param name="activeColsList">The list of active columns for each image.</param>
        /// <param name="trainingImages">The training images used during the experiment.</param>
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

        /// <summary>
        /// Draws the similarity plot based on the highest similarity scores for the images.
        /// The plot is saved as a PNG file in the current directory under "SimilarityPlots" folder.
        /// </summary>
        /// <param name="highestSimilarities">The list of highest similarity values to plot.</param>
        public static void DrawSimilarityPlots(List<double> highestSimilarities)
        {
            if (highestSimilarities == null || highestSimilarities.Count == 0)
            {
                Debug.WriteLine("No similarity data available.");
                return;
            }

            // Define the folder path based on the current directory
            string folderPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots");

            // Create the folder if it doesn't exist
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Define the file name
            string fileName = "similarity_plot.png";

            // Define the file path with the folder path and file name
            string filePath = Path.Combine(folderPath, fileName);

            // Draw the similarity plot for the highest similarity scores
            NeoCortexUtils.DrawCombinedSimilarityPlot(highestSimilarities, filePath, 800, 1200);

            Debug.WriteLine($"FilePath: {filePath}");
            Debug.WriteLine("Similarity plot generated and saved successfully.");
        }

        /// <summary>
        /// Generates heatmap matrices for each image based on the provided heatmap data.
        /// Each heatmap is saved as a PNG file in the current directory under "HeatMapMatricsforImage" folder.
        /// </summary>
        /// <param name="heatmapData">The heatmap data representing the image matrix values.</param>
        private void GenerateMatricsforImage(List<List<double>> heatmapData)
        {
            int i = 1;

            foreach (var values in heatmapData)
            {
                // Define the folder path from the current Directory
                string folderPath = Path.Combine(Environment.CurrentDirectory, "HeatMapMatricsforImage");

                // Create the folder if it doesn't exist
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Define the file path with the folder path
                string filePath = Path.Combine(folderPath, $"heatmapMatrix_{i}.png");

                // Debugging the FilePath
                Debug.WriteLine($"FilePath: {filePath}");

                // Assuming the input data should be in a 52x52 matrix (rows x columns)
                int rows = 52;
                int cols = 52;

                // Check if the number of values matches the expected size (52x52)
                if (values.Count != rows * cols)
                {
                    Debug.WriteLine("Data does not match expected size of Image Height and Width.");
                    continue;  // Skip this row if data doesn't match
                }

                // Create a heatmap for the data
                NeoCortexUtils.SaveHeatmapValuesAsImage(values, filePath, rows, cols, 50);

                // Debugging the Message
                Debug.WriteLine($"Heatmap values {i} generated and saved successfully.");

                i++;
            }
        }
    }
}
