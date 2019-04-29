using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace MulticlassClassification
{
    public class ModelBuilder
    {
        private static string TrainDataPath = GetAbsolutePath("./Data/TrainData.txt");
        private static string TestDataPath = GetAbsolutePath("./Data/TestData.txt");
        public ModelBuilder()
        {
        }

        public EstimatorChain<ColumnConcatenatingTransformer> DataSetup(MLContext context)
        {
            // loads the data
            var trainingDataView = context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            var testDataView = context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);

            var outputDataSetup = context
                .Transforms
                .Conversion
                .MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(IrisData.Label));
            var inputDataSetup = context.Transforms.Concatenate(
                    "Features",
                    nameof(IrisData.SepalLength),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLength),
                    nameof(IrisData.PetalWidth)
                    );
            var dataSetup = outputDataSetup.Append(inputDataSetup).AppendCacheCheckpoint(context);
            return dataSetup;
        }

        public void TrainModel(MLContext context)
        {
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "KeyColumn",
                                                                                       featureColumnName: "Features");
        }

        public static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

    }
}
