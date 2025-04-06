using NeoCortex;
using NeoCortexApi;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using NeoCortexApi.Utility;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.IO;

namespace NeoCortexApiSample
{
    /// <summary>
    /// Implements an experiment that demonstrates how to learn spatial patterns.
    /// SP will learn every presented input in multiple iterations.
    /// </summary>
    public class SpatialPatternLearning
    {
        /// <summary>
        /// Main function that runs the experiment.
        /// It initializes the HTM configuration, encoder, and spatial pooler, 
        /// and trains the model on spatial patterns.
        /// </summary>
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SpatialPatternLearning)}");

            double minOctOverlapCycles = 1.0;
            double maxBoost = 5.0;
            int inputBits = 200;
            int numColumns = 1024;

            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                CellsPerColumn = 10,
                MaxBoost = maxBoost,
                DutyCyclePeriod = 100,
                MinPctOverlapDutyCycles = minOctOverlapCycles,
                GlobalInhibition = false,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                LocalAreaDensity = -1,
                ActivationThreshold = 10,
                MaxSynapsesPerSegment = (int)(0.01 * numColumns),
                Random = new ThreadSafeRandom(42),
                StimulusThreshold = 10,
            };

            double max = 100;

            Dictionary<string, object> settings = new Dictionary<string, object>()
            {
                { "W", 15},
                { "N", inputBits},
                { "Radius", -1.0},
                { "MinVal", 0.0},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
                { "MaxVal", max}
            };

            EncoderBase encoder = new ScalarEncoder(settings);

            List<double> inputValues = new List<double>();
            for (int i = 0; i < (int)max; i++)
            {
                inputValues.Add((double)i);
            }

            var sp = RunExperiment(cfg, encoder, inputValues);
            RunRustructuringExperiment(sp, encoder, inputValues);
        }

        /// <summary>
        /// Runs the core experiment for training the Spatial Pooler (SP).
        /// This function initializes the HTM memory, the spatial pooler, and runs multiple learning cycles.
        /// </summary>
        private static SpatialPooler RunExperiment(HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            var mem = new Connections(cfg);
            bool isInStableState = false;
            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, inputValues.Count * 40,
                (isStable, numPatterns, actColAvg, seenInputs) =>
                {
                    if (!isStable)
                    {
                        Debug.WriteLine($"INSTABLE STATE");
                        isInStableState = false;
                    }
                    else
                    {
                        Debug.WriteLine($"STABLE STATE");
                        isInStableState = true;
                    }
                });

            SpatialPooler sp = new SpatialPooler(hpa);
            sp.Init(mem, new DistributedMemory() { ColumnDictionary = new InMemoryDistributedDictionary<int, NeoCortexApi.Entities.Column>(1) });

            CortexLayer<object, object> cortexLayer = new CortexLayer<object, object>("L1");
            cortexLayer.HtmModules.Add("encoder", encoder);
            cortexLayer.HtmModules.Add("sp", sp);

            double[] inputs = inputValues.ToArray();
            Dictionary<double, int[]> prevActiveCols = new Dictionary<double, int[]>();
            Dictionary<double, double> prevSimilarity = new Dictionary<double, double>();

            foreach (var input in inputs)
            {
                prevSimilarity.Add(input, 0.0);
                prevActiveCols.Add(input, new int[0]);
            }

            int maxSPLearningCycles = 1;
            int numStableCycles = 0;

            for (int cycle = 0; cycle < maxSPLearningCycles; cycle++)
            {
                Debug.WriteLine($"Cycle  * {cycle} * Stability: {isInStableState}");

                foreach (var input in inputs)
                {
                    double similarity;

                    var lyrOut = cortexLayer.Compute((object)input, true) as int[];
                    var activeColumns = cortexLayer.GetResult("sp") as int[];
                    var actCols = activeColumns.OrderBy(c => c).ToArray();

                    similarity = MathHelpers.CalcArraySimilarity(activeColumns, prevActiveCols[input]);

                    Debug.WriteLine($"[cycle={cycle.ToString("D4")}, i={input}, cols=:{actCols.Length} s={similarity}] SDR: {Helpers.StringifyVector(actCols)}");

                    prevActiveCols[input] = activeColumns;
                    prevSimilarity[input] = similarity;
                }

                if (isInStableState)
                {
                    numStableCycles++;
                }

                if (numStableCycles > 5)
                    break;
            }

            return sp;
        }

        /// <summary>
        /// Executes a restructuring experiment to analyze the behavior of the spatial pooler.
        /// It computes the reconstruction of the SDR and compares it to the original input SDR.
        /// </summary>
        private void RunRustructuringExperiment(SpatialPooler sp, EncoderBase encoder, List<double> inputValues)
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
            GenerateHeatmap(heatmapData);
            GenerateMatrics(heatmapData);
            GenerateEncodedMatrics(encodedInputs.Select(arr => arr.ToList()).ToList());
            GenerateReconstructedMatrics(normalizedPermanence.Select(arr => arr.ToList()).ToList());
            DrawSimilarityPlots(similarityList);

            Console.WriteLine("All heatmaps generated and similarity plots saved.");
        }

        /// <summary>
        /// Generates heatmap image from the collected experiment data.
        /// </summary>

        private void GenerateHeatmap(List<List<double>> heatmapData)
        {
            int i = 1;
            foreach (var values in heatmapData)
            {
                string folderPath = Path.Combine(Environment.CurrentDirectory, "HeatMap");
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                string filePath = Path.Combine(folderPath, $"heatmap-{i}.png");

                Debug.WriteLine($"FilePath: {filePath}");

                int rows = 8;
                int cols = 25;

                if (values.Count != rows * cols)
                {
                    Debug.WriteLine("Data does not match expected size of 8x25.");
                    continue;
                }

                NeoCortexUtils.DrawBitHeatmap(values, filePath, rows, cols, 50);

                Debug.WriteLine($"Heatmap {i} generated and saved successfully.");
                i++;
            }
        }

        /// <summary>
        /// Generates heatmap matrices from the collected experiment data.
        /// </summary>
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

        /// <summary>
        /// Generates encoded input matrices from the SDRs.
        /// </summary>
        private void GenerateEncodedMatrics(List<List<int>> encodedInputs)
        {
            int i = 1;

            foreach (var inputs in encodedInputs)

            {
                // Define the folder path from the current Directory
                string folderPath = Path.Combine(Environment.CurrentDirectory, "EncodedInputMatrics");
                // Create the folder if it doesn't exist
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Define the file path with the folder path
                string filePath = Path.Combine(folderPath, $"EncodedInputMatrics_{i}.png");

                // Debugging the FilePath
                Debug.WriteLine($"FilePath: {filePath}");

                // Assuming the input data should be in an 8x25 matrix (rows x columns)
                // Convert the current row to a 2D array (8x25) directly
                int rows = 8;
                int cols = 25;

                // Check if the number of values matches the expected size (8x25)
                if (inputs.Count != rows * cols)
                {
                    Debug.WriteLine("Data does not match expected size of 8x25.");
                    continue;  // Skip this row if data doesn't match
                }

                // Create a heatmap for the data
                NeoCortexUtils.SaveInputValuesAsImage(inputs, filePath, rows, cols, 50);

                // Debugging the Message
                Debug.WriteLine($"Encoded Matrix {i} generated and saved successfully.");

                i++;
            }
        }

        /// <summary>
        /// Generates and saves reconstructed matrices from the normalized permanence values.
        /// </summary>
        private void GenerateReconstructedMatrics(List<List<int>> normalizedPermanence)
        {
            int i = 1;

            foreach (var reconstructedInputs in normalizedPermanence)

            {
                // Define the folder path from the current Directory
                string folderPath = Path.Combine(Environment.CurrentDirectory, "NumericReconstrucedInputMatrics");
                // Create the folder if it doesn't exist
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Define the file path with the folder path
                string filePath = Path.Combine(folderPath, $"NumericReconstructedInputMatrics_{i}.png");

                // Debugging the FilePath
                Debug.WriteLine($"FilePath: {filePath}");

                // Assuming the input data should be in an 8x25 matrix (rows x columns)
                // Convert the current row to a 2D array (8x25) directly
                int rows = 8;
                int cols = 25;

                // Check if the number of values matches the expected size (8x25)
                if (reconstructedInputs.Count != rows * cols)
                {
                    Debug.WriteLine("Data does not match expected size of 8x25.");
                    continue;  // Skip this row if data doesn't match
                }

                // Create a heatmap for the data
                NeoCortexUtils.SaveInputValuesAsImage(reconstructedInputs, filePath, rows, cols, 50);

                // Debugging the Message
                Debug.WriteLine($"Encoded Matrix {i} generated and saved successfully.");

                i++;
            }
        }

        /// <summary>
        /// Generates and saves similarity plots for the experiment.
        /// </summary>
        public static void DrawSimilarityPlots(List<double[]> similaritiesList)
        {
            // Combine all similarities from the list of arrays
            List<double> combinedSimilarities = new List<double>();
            foreach (var similarities in similaritiesList)
            {
                combinedSimilarities.AddRange(similarities);
            }

            // Define the folder path based on the current directory
            string folderPath = Path.Combine(Environment.CurrentDirectory, "SimilarityPlots");

            // Create the folder if it doesn't exist
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            // Define the file name
            string fileName = "combined_similarity_plot.png";

            // Define the file path with the folder path and file name
            string filePath = Path.Combine(folderPath, fileName);

            // Draw the combined similarity plot
            NeoCortexUtils.DrawCombinedSimilarityPlot(combinedSimilarities, filePath, 4500, 1100);
            //Debugging the Filepath
            Debug.WriteLine($"FilePath: {filePath}");

            Debug.WriteLine($"Combined similarity plot generated and saved successfully.");
        }
    }
}
