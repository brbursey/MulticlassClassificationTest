using System.Linq;
using MulticlassClassification.Repositories;

namespace MulticlassClassification
{
    public class Program
    {    
        public static void Main(string[] args)
        {
            IIrisDataParser dataParser = new IrisDataParser();
            IDataProvider provider = new IrisDataProvider();
            IIrisDataRepository dataRepository = new IrisDataRepository(dataParser);
            
            var classifier = new ClassificationModel(provider);
            var prediction = new IrisDataPrediction(dataRepository);
            
//            var data = prediction.GetData();
//            var probs = prediction.GetProbabilities();
           
            classifier.CreateDirectoryAndExtractZipfile(provider.BaseModelPath, provider.ModelZipFilePath);
            classifier.FitAndSaveModel();
            prediction.Probabilities = classifier.PredictValues(prediction.Data, prediction.Categories);
        }
    }
}
