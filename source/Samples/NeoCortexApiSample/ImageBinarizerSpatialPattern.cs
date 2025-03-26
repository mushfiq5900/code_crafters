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
        public string inputPrefix { get; private set; }

        /// <summary>
        /// Implements an experiment that demonstrates how to learn spatial patterns.
        /// SP will learn every presented Image input in multiple iterations.
        /// </summary>
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(ImageBinarizerSpatialPattern)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int numColumns = 84 * 84;
            int imageSize = 52;
            var colDims = new int[] { 84, 84 };

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
            // Run Experiment & get active columns per image
            var (sp, activeColumnsPerImage) = RunExperiment(cfg, inputPrefix);

            // Run Reconstruction using precomputed active columns
            RunRustructuringExperiment(sp, activeColumnsPerImage, trainingImages);
        }


        /// <summary>
        /// Implements the experiment.
        /// </summary>
        /// <param name="cfg"></param>
        /// <param name="inputPrefix"> The name of the images</param>
        /// <returns>The trained bersion of the SP.</returns>
        private (SpatialPooler, List<int[]>) RunExperiment(HtmConfig cfg, string inputPrefix)
        {
            Debug.WriteLine("[INFO] Initializing Experiment...");
            var mem = new Connections(cfg);
            bool isInStableState = false;
            int numColumns = 84 * 84; // SDR size

            string trainingFolder = "Sample\\TestFiles";  // Folder with images
            string binarizedFolder = "Binarized";         // Folder to save binarized images
            Directory.CreateDirectory(binarizedFolder);

            var trainingImages = Directory.GetFiles(trainingFolder, $"{inputPrefix}*.jpeg");
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

            while (!isInStableState && currentCycle < maxCycles)
            {
                Debug.WriteLine($"Processing Training Cycle for each Image {currentCycle}");

                foreach (var image in trainingImages)
                {
                    // *1. Binarize Image and Save*
                    string binarizedFile = Path.Combine(binarizedFolder, $"{Path.GetFileNameWithoutExtension(image)}.txt");
                    if (!File.Exists(binarizedFile)) // Avoid reprocessing
                    {
                        string generatedFile = NeoCortexUtils.BinarizeImage(image, imgSize, testName);
                        File.Copy(generatedFile, binarizedFile, true);
                    }

                    // *2. Read Binarized Image as Input Vector*
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

            // *3. Pass Active Columns to Reconstruction Experiment*
            RunRustructuringExperiment(sp, activeColsList, trainingImages);


            return (sp, activeColsList);
        }





        private void RunRustructuringExperiment(SpatialPooler sp, List<int[]> activeColsList, string[] trainingImages)
        {
            List<int[]> normalizedPermanence = new List<int[]>();
            List<double[]> similarityList = new List<double[]>();
            foreach (var actcols in activeColsList)
            {
                Debug.WriteLine("Reconstructing permanence for SDR...");

                // Reconstruct the permanence for the predicted SDR
                Dictionary<int, double> reconstructedPermanence = sp.Reconstruct(actcols);

                Dictionary<int, double> allPermanenceDictionary = new Dictionary<int, double>();
                foreach (var kvp in reconstructedPermanence)
                {
                    //Debug.WriteLine($"Index: {kvp.Key}, Permanence Value: {kvp.Value}");
                    allPermanenceDictionary[kvp.Key] = kvp.Value;
                }

                int imgsize = 52 * 52;

                // Assign inactive columns permanence 0
                for (int inputIndex = 0; inputIndex < imgsize; inputIndex++)
                {
                    if (!reconstructedPermanence.ContainsKey(inputIndex))
                    {
                        allPermanenceDictionary[inputIndex] = 0.0;
                    }
                }

                // Normalize permanence values
                var ThresholdValue = 67.0;
                List<double> permanenceValuesList = allPermanenceDictionary.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
                Debug.WriteLine($"[INFO] Applying Threshold for Nomalizing the Permanence Values");
                List<int> normalizePermanenceList = Helpers.ThresholdingProbabilities(permanenceValuesList, ThresholdValue);

                normalizedPermanence.Add(normalizePermanenceList.ToArray());

                // Define a unique image index for consistency
                int imageIndex = activeColsList.IndexOf(actcols);

                // Generate consistent names for both images
                string reconstructedImageName = $"ReconstructedImage_{imageIndex}";
                string heatmapImageName = $"Heatmap_{imageIndex}";

                // Save the reconstructed binary image
                NeoCortexUtils.GenarateReconstrucetedBinarizedImage(normalizePermanenceList.ToArray(), reconstructedImageName);

                // *Save heatmap per cycle*
                List<List<double>> heatmapData = new List<List<double>> { permanenceValuesList }; // Use sorted permanence
                string folderPath = Path.Combine(Directory.GetCurrentDirectory(), "Heatmaps");
                Directory.CreateDirectory(folderPath);
                string heatmapFilePath = Path.Combine(folderPath, $"{heatmapImageName}.png");

                NeoCortexUtils.GenarateImageInputHeatmap(heatmapData, heatmapFilePath);


                for (int i = 0; i < trainingImages.Length; i++)
                {
                    string imageName = Path.GetFileNameWithoutExtension(trainingImages[i]);  // Extract "1" from "1.jpg"
                    string binarizedFile = Path.Combine("Binarized", $"{imageName}.txt");    // Look for "Binarized/1.txt"

                    if (File.Exists(binarizedFile))
                    {
                        // Read the binarized image (input vector)
                        int[] inputVector = NeoCortexUtils.ReadCsvIntegers(binarizedFile).ToArray();

                        // Ensure we have the correct normalized permanence for this image
                        if (i < normalizedPermanence.Count)
                        {
                            int[] currentNormalizedPermanence = normalizedPermanence[i];

                            // Compute Jaccard Similarity
                            double jaccardSim = MathHelpers.JaccardSimilarityofBinaryArrays(inputVector, currentNormalizedPermanence);

                            // Store similarity result for plotting
                            similarityList.Add(new double[] { jaccardSim });

                            Debug.WriteLine($"Image {imageName}.jpg | Jaccard Similarity: {jaccardSim}");
                        }
                        else
                        {
                            Debug.WriteLine($"Warning: No normalized permanence found for {imageName}.jpg");
                        }
                    }
                    else
                    {
                        Debug.WriteLine($"Warning: Binarized file {binarizedFile} not found.");
                    }
                }

            }

            // After processing all images, plot the similarity results
            DrawSimilarityPlots(similarityList);

        }

        public static void DrawSimilarityPlots(List<double[]> similaritiesList)
        {
            // Ensure there is at least one cycle to process
            if (similaritiesList == null || similaritiesList.Count == 0)
            {
                Debug.WriteLine("No similarity data available.");
                return;
            }

            // Get only the last cycle's similarity values (multiple images)
            List<double> lastCycleSimilarities = new List<double>();

            foreach (var similarity in similaritiesList)
            {
                lastCycleSimilarities.AddRange(similarity); // Collect all similarities from the last cycle
            }

            // Define the folder path based on the current directory
            string folderPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots");

            // Create the folder if it doesn't exist
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Define the file name
            string fileName = "last_cycle_Image_similarity_plot.png";

            // Define the file path with the folder path and file name
            string filePath = Path.Combine(folderPath, fileName);

            // Draw the similarity plot for the last cycle
            NeoCortexUtils.DrawCombinedSimilarityPlot(lastCycleSimilarities, filePath, 4500, 1100);

            // Debugging the Filepath
            Debug.WriteLine($"FilePath: {filePath}");
            Debug.WriteLine("Last cycle similarity plot generated and saved successfully.");
        }


    }
}