﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MulticlassClassification.DataStructures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace MulticlassClassification.ClassificationModel
{
    public class ClassificationModel
    {
        private readonly IDataProvider provider;
        public readonly MLContext Context;
        
        private readonly IDataView TrainingDataView;
        private readonly IDataView TestDataView;
        private readonly IModelBuilder ModelBuilder;

        public readonly EstimatorChain<KeyToValueMappingTransformer> Trainer;
        public readonly EstimatorChain<ColumnConcatenatingTransformer> Pipeline;

        public ClassificationModel(IDataProvider provider)
        {
            this.provider = provider;
            Context = provider.Context;
            TrainingDataView = provider.TrainingDataView;
            TestDataView = provider.TestDataView;
            ModelBuilder = provider.ModelBuilder;
            Trainer = ModelBuilder.CreateTrainerForModel(Context);
            Pipeline = ModelBuilder.DataPipelineSetup(Context);

        }
        
        public void FitAndSaveModel()
        {
            var stopwatch = Stopwatch.StartNew();
            var pipeline = Pipeline.Append(Trainer);
            var trainedModel = pipeline.Fit(TrainingDataView);
            stopwatch.Stop();
            long elapsedMs = stopwatch.ElapsedMilliseconds;
            Console.WriteLine($"***** Training time: {elapsedMs / 1000} seconds *****");

            var predictions = trainedModel.Transform(TestDataView);
            var metrics = Context.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            ConsoleHelper.PrintMultiClassClassificationMetrics(Trainer.ToString(), metrics);

            Context.Model.Save(trainedModel,
                               TrainingDataView.Schema,
                               provider.ModelPath);

            Console.WriteLine("The model is saved to {0}", provider.ModelPath);
        }

        public IEnumerable<Dictionary<string, float>> PredictValues(IEnumerable<IrisData> dataToPredict, IEnumerable<string> dataCategories)
        {
            var trainedMulticlassModel = Context.Model.Load(provider.ModelPath, out var modelInputSchema);
            var categories = OutputCategories(dataCategories);

            var predEngine = Context.Model.CreatePredictionEngine<IrisData, IrisPrediction>(trainedMulticlassModel);
            VBuffer<float> keys = default;
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();

            var probabilities = new List<Dictionary<string, float>>();
            foreach (var data in dataToPredict)
            {
                var resultPrediction = predEngine.Predict(data);
                var probabilitiesByLabel = new Dictionary<string, float>();
                for (var i = 0; i < labelsArray.Length; i++)
                {
                    probabilitiesByLabel.Add(categories[labelsArray[i]], resultPrediction.Score[i]);
                }
                probabilities.Add(probabilitiesByLabel);
            }
            return probabilities;
        }

        

        private Dictionary<float, string> OutputCategories(IEnumerable<string> categories)
        {
            var indexToCategory = new Dictionary<float, string>();
            var index = 0;
            foreach (var category in categories)
            {
                indexToCategory.Add(index, category);
                index = index + 1;
            }
            return indexToCategory;
        }
        
//        private void PredictTestValues()
//        {
//            var categories = new List<string>()
//            {
//                "Setosa",
//                "Virginica",
//                "Versicolor"
//            };
//            var IrisFlowers = OutputCategories(categories);
//
//            var predEngine = Context.Model.CreatePredictionEngine<IrisData, Prediction>(TrainedMulticlassModel);
//            VBuffer<float> keys = default;
//            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
//            var labelsArray = keys.DenseValues().ToArray();
//            Console.WriteLine("=====Predicting using model====");
//
//            var resultPrediction1 = predEngine.Predict(SampleIrisData.Iris1);
//
//            Console.WriteLine($"Actual: Setosa.\n" +
//                              $"Predicted label and score:\n" +
//                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction1.Score[0]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction1.Score[1]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction1.Score[2]:0.####}\n");
//
//            var resultPrediction2 = predEngine.Predict(SampleIrisData.Iris2);
//
//            Console.WriteLine($"Actual: Virginica.\n" +
//                              $"Predicted label and score:\n" +
//                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction2.Score[0]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction2.Score[1]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction2.Score[2]:0.####}\n");
//
//            var resultPrediction3 = predEngine.Predict(SampleIrisData.Iris3);
//
//            Console.WriteLine($"Actual: Versicolor.\n" +
//                              $"Predicted label and score:\n" +
//                              $"{IrisFlowers[labelsArray[0]]}: {resultPrediction3.Score[0]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[1]]}: {resultPrediction3.Score[1]:0.####}\n" +
//                              $"{IrisFlowers[labelsArray[2]]}: {resultPrediction3.Score[2]:0.####}\n");
//        }
    }
}
