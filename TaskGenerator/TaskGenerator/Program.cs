using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using QuikGraph;

class TaskGenerator
{
    static float skew = 2f;
    static float wDag = 20f; //average computational cost of the graph

    static int taskCount;
    static int[] possibleTaskCounts = new[] {30,40,50,60,70,80,100};
    static float shape;
    static float[] possibleShapes = new[] {0.5f,1.0f,2.0f};
    static int outDegree;
    static int[] possibleOutDegrees = new[] {1,2,3,4,5};
    static float comCompRatio;
    static float[] possibleComCompRatio = new[] {0.1f,0.5f,1.0f,5.0f,10.0f};
    static float compCostRange;
    static float[] possibleCompCostRange = new[] {0.1f,0.5f,1.0f};
    static int processorCount;
    static int[] possibleProcessorCount = new[] {4,8,12,16,20};
     
    static int height, width;

    static AdjacencyGraph<int, IEdge<int>> graph;
    static int[] levelForTask;
    static int[,] costForTaskInProcessor;

    static readonly Random random = new();

    static void Main(string[] args)
    {
        ParseInput(args);

        InitGraphWithVetexes();

        SubdivideVertexesInLevels();

        CreateEdges();

        ComputeCostsForEachTaskInProcessors();

        PrintInfo();
    }

    static void ComputeCostsForEachTaskInProcessors()
    {
        costForTaskInProcessor = new int[taskCount, processorCount];
        var uniformCosts = new ContinuousUniform(0, 2 * wDag);

        var averageCostForTask = uniformCosts.Sample();

        for (int i = 0; i < costForTaskInProcessor.GetLength(0); i++)
        {
            for (int j = 0; j < costForTaskInProcessor.GetLength(1); j++)
            {
                costForTaskInProcessor[i, j] = random.Next((int)(averageCostForTask * (1 - compCostRange / 2)), (int)(averageCostForTask * (1 + compCostRange / 2)));
            }
        }


        //print costs info
        for (int i = 0; i < costForTaskInProcessor.GetLength(0); i++)
        {
            Console.Write($"{i} ==> ");
            for (int j = 0; j < costForTaskInProcessor.GetLength(1); j++)
            {
                Console.Write($"{costForTaskInProcessor[i, j]}, ");
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    static void CreateEdges()
    {
        for (int i = 0; i < taskCount-1; i++)
        {
            var destinations = random.NextInt32Sequence(i+1, taskCount).Take(outDegree);
            foreach (var node in destinations)
            {
                if (levelForTask[node] > levelForTask[i] && !graph.ContainsEdge(i, node))
                {
                    graph.AddEdge(new Edge<int>(i, node));
                }
            }
        }

        //print edges info
        var count = 0;
        for (int i = 0; i < taskCount; i++)
        {
            Console.Write($"{i} ==> ");
            var linkedNodes = graph.Edges.Where(e => e.Source == i).Select(e=>e.Target);
            Console.Write(string.Join(", ", linkedNodes));
            Console.WriteLine();

            count += linkedNodes.Count(); 
        }
        Console.WriteLine();

        var mean = (float)count / taskCount;
        Console.WriteLine($"mean OutDegree espected: {outDegree}, actual {mean}\n");
    }


    static void InitGraphWithVetexes()
    {
        graph = new AdjacencyGraph<int, IEdge<int>>(false, taskCount);

        for (int i = 0; i < taskCount; i++)
        {
            graph.AddVertex(i); //id del nodo
        }
    }

    static void SubdivideVertexesInLevels()
    {
        var heightMean = Math.Sqrt(taskCount) / shape;
        var widthMean = shape * Math.Sqrt(taskCount);
        var uniformHeight = new ContinuousUniform(heightMean / skew, 2 * heightMean - heightMean / skew);
        var uniformWidth = new ContinuousUniform(widthMean / skew, 2 * widthMean - widthMean / skew);

        height = (int)Math.Ceiling(uniformHeight.Sample());

        int tasksAlreadyInALevel = 0;
        levelForTask = new int[taskCount];
        for (int i = 0; i < taskCount; i++)
        {
            levelForTask[i] = -1; //TODO: cosa fare con i nodi che non vengono collocati in nessun livello?
        }

        for (int actualLevel = 0; actualLevel < height; actualLevel++)
        {
            if (tasksAlreadyInALevel < taskCount)
            {
                width = (int)Math.Ceiling(uniformWidth.Sample());
                //aggiungo width task al livello attuale.
                var taskToAdd = Math.Min(taskCount - tasksAlreadyInALevel, width);
                for (int j = 0; j < taskToAdd; j++)
                {
                    levelForTask[tasksAlreadyInALevel++] = actualLevel;
                }
            }
        }

        Console.WriteLine($"height mean: {uniformHeight.Mean} == {heightMean}");
        Console.WriteLine(uniformHeight.ToString());
        Console.WriteLine($"width mean: {uniformWidth.Mean} == {widthMean}");
        Console.WriteLine(uniformWidth.ToString());

        Console.WriteLine();
        Console.WriteLine("Livello per ogni task: ");
        for (int i = 0; i < levelForTask.Length; i++)
        {
            Console.Write(levelForTask[i]);
            Console.Write(" ");
        }
        Console.WriteLine();
    }

    static void PrintInfo()
    {
        Console.WriteLine();
        Console.WriteLine("Values: {0}: {1} | {2}: {3} | {4}: {5} | {6}: {7} | {8}: {9} | {10}: {11}| {12}: {13}",
            nameof(taskCount), taskCount, nameof(shape), shape, nameof(outDegree), outDegree, nameof(comCompRatio), comCompRatio, nameof(compCostRange), compCostRange, nameof(height), height, nameof(processorCount), processorCount);
        Console.WriteLine();
    }

    static void ParseInput(string[] args)
    {
        if (args.Length > 0)
        {
            if(args.Length == 1)
            {
                taskCount = possibleTaskCounts[random.Next(0, possibleTaskCounts.Length)];
                shape = possibleShapes[random.Next(0, possibleShapes.Length)];
                outDegree = possibleOutDegrees[random.Next(0, possibleOutDegrees.Length)];
                comCompRatio = possibleComCompRatio[random.Next(0, possibleComCompRatio.Length)];
                compCostRange = possibleCompCostRange[random.Next(0, possibleCompCostRange.Length)];
                processorCount = possibleProcessorCount[random.Next(0, possibleProcessorCount.Length)];
            }
            else if (args.Length != 6)
            {
                Console.WriteLine("Usage: TaskGenerator [[random] | [{0} {1} {2} {3} {4} {5}]]", nameof(taskCount), nameof(shape), nameof(outDegree), nameof(comCompRatio), nameof(compCostRange), nameof(processorCount));
                throw new ArgumentException(nameof(args));
            }
            else
            {
                try
                {
                    taskCount = int.Parse(args[0]);
                    shape = float.Parse(args[1]);
                    outDegree = int.Parse(args[2]);
                    comCompRatio = float.Parse(args[3]);
                    compCostRange = float.Parse(args[4]);
                    processorCount = int.Parse(args[5]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex.Message);
                    throw;
                }
            }
        }
        else
        {
            do { Console.WriteLine($"{nameof(taskCount)}: "); } while (!int.TryParse(Console.ReadLine() ?? string.Empty, out taskCount));
            do { Console.WriteLine($"{nameof(shape)}: "); } while (!float.TryParse(Console.ReadLine() ?? string.Empty, out shape));
            do { Console.WriteLine($"{nameof(outDegree)}: "); } while (!int.TryParse(Console.ReadLine() ?? string.Empty, out outDegree));
            do { Console.WriteLine($"{nameof(comCompRatio)}: "); } while (!float.TryParse(Console.ReadLine() ?? string.Empty, out comCompRatio));
            do { Console.WriteLine($"{nameof(compCostRange)}: "); } while (!float.TryParse(Console.ReadLine() ?? string.Empty, out compCostRange));
            do { Console.WriteLine($"{nameof(processorCount)}: "); } while (!int.TryParse(Console.ReadLine() ?? string.Empty, out processorCount));
        }
    }
}