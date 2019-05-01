using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace MulticlassClassification.DataStructures
{
     public class IrisData
        {
            [LoadColumn(0)]
            public float Label;
    
            [LoadColumn(1)]
            public float SepalLength;
    
            [LoadColumn(2)]
            public float SepalWidth;
    
            [LoadColumn(3)]
            public float PetalLength;
    
            [LoadColumn(4)]
            public float PetalWidth;

            public Dictionary<string, float> Probabilities { get; set; }
    }
}
