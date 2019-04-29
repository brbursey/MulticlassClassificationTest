using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Transforms;
using MulticlassClassification.DataStructures;

namespace MulticlassClassification
{
    public class ModelBuilder
    {
        public ModelBuilder()
        {
        }

        public EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> TrainingModelSetup(MLContext context)
        {
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
            var dataPipeline = outputDataSetup.Append(inputDataSetup).AppendCacheCheckpoint(context);
            
            var trainer = Trainer(context);
            var trainingPipeline = dataPipeline.Append(trainer);
            return trainingPipeline;

        }

        public EstimatorChain<KeyToValueMappingTransformer> Trainer(MLContext context)
        {
            var trainer = context
                .MulticlassClassification
                .Trainers
                .SdcaMaximumEntropy(
                    labelColumnName: "KeyColumn", 
                    featureColumnName: "Features")
                .Append(
                    context
                        .Transforms
                        .Conversion
                        .MapKeyToValue(
                            outputColumnName: nameof(IrisData.Label), 
                            inputColumnName: "KeyColumn"));
            return trainer;
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
