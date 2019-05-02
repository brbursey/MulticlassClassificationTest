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
            var prediction = new IrisDataPrediction(dataRepository, provider);
           
            classifier.CreateDirectoryAndExtractZipfile(provider.BaseModelPath, provider.ModelZipFilePath);
            classifier.FitAndSaveModel();
            classifier.PredictValues(prediction.Data, prediction.Categories);
        }
    }
}
