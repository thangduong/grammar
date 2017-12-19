using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TokenizerCS;

namespace tokenizercsTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Tokenizer tok = new Tokenizer();
            string input_string = "F.D.R. was a great president in the 1930s, and A.I.G was an insurance company. 1929 was a bad year. bob's cat is black. don't do it.";
            var result = tok.Tokenize(input_string);
            System.Console.WriteLine(input_string);
            foreach (var token in result)
            {
                System.Console.WriteLine("{0}, {1}, {2}, {3}", token.token, token.start, token.len, token.type);
            }
        }
    }
}
