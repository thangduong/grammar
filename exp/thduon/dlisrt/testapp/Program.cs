using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OXOGrammarModels;
using Newtonsoft.Json;
using System.IO;

namespace testapp
{
    class Program
    {
        static void Main(string[] args)
        {
            CommaModel m = new CommaModel();
            m.Initialize();
            string result = m.Execute("{sentence:\"<pad> <pad> <pad> <pad> <S> this is a test sentence and that is not a test sentence <pad> <pad> <pad>\"}");
            System.Console.WriteLine(result);
        }
    }
}
