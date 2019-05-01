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
    public interface IModelBuilder
    {
        EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> TrainingModelSetup(MLContext context);
        EstimatorChain<KeyToValueMappingTransformer> CreateTrainerForModel(MLContext context);
        void TrainModel(MLContext context);
    }
    public class IrisModelBuilder : IModelBuilder
    {
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
            
            //TODO: This dude is run twice. FIX IT!
            var trainer = CreateTrainerForModel(context);
            var trainingPipeline = dataPipeline.Append(trainer);
            return trainingPipeline;
        }

        public EstimatorChain<KeyToValueMappingTransformer> CreateTrainerForModel(MLContext context)
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
            var trainer = context
                .MulticlassClassification
                .Trainers
                .NaiveBayes(labelColumnName: "KeyColumn", featureColumnName: "Features");
        }
    }
}
