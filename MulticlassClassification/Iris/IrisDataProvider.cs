using System.IO;
using Microsoft.ML;
using MulticlassClassification.ClassificationModel;
using MulticlassClassification.Iris.DataStructures;

namespace MulticlassClassification.Iris
{
    public interface IDataProvider
    {
        MLContext Context { get; set; }
        string BaseModelPath { get; set; }
        string ModelZipFilePath { get; set; }
        string ModelPath { get; set; }
        IDataView TrainingDataView { get; set; }
        IDataView TestDataView { get; set; }
        IModelBuilder ModelBuilder { get; set; }

        void CreateDirectoryAndExtractZipfile(string providerBaseModelPath, string providerModelZipFilePath);
    }
    public class IrisDataProvider : IDataProvider
    {
        public MLContext Context { get; set; }
        public string BaseModelPath { get; set; }
        public string ModelZipFilePath { get; set; }
        public string ModelPath { get; set; }
        public string TrainDataPath { get; set; }
        public string TestDataPath { get; set; }
        public IDataView TrainingDataView { get; set; }
        public IDataView TestDataView { get; set; }
        public IModelBuilder ModelBuilder { get; set; }

        public IrisDataProvider()
        {
            Context = new MLContext(seed: 0);
            BaseModelPath = ".\\MLModels";
            ModelZipFilePath = GetAbsolutePath($"{BaseModelPath}\\TestClassificationModel.zip");
            ModelPath = GetAbsolutePath($"{BaseModelPath}\\TestClassificationModel.zip");
            TrainDataPath = ".\\Data\\TrainData.txt";
            TestDataPath = ".\\Data\\TestData.txt";
            TrainingDataView = Context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            TestDataView = Context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);
            ModelBuilder = new IrisModelBuilder();
        }

        private static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            var assemblyFolderPath = dataRoot.Directory.FullName;
            var fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
        
        // needs some or a lot of work
        public void CreateDirectoryAndExtractZipfile(string dirPath, string zipfileLocation)
        {
            Directory.CreateDirectory(dirPath);
            
//            if (!File.Exists(zipfileLocation))
//            {
//                ZipFile.CreateFromDirectory(dirPath, zipfileLocation);   
//            }
            //ZipFile.ExtractToDirectory(zipfileLocation, dirPath);
        }
    }
}