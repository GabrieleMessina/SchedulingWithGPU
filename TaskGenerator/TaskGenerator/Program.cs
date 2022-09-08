using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using QuikGraph;
using System.Text;

namespace TaskGenerator;

public static class TaskGenerator
{
    static readonly Random random = new();
    static readonly float skew = 2f;
    static readonly float wDag = 20f; //average computational cost of the graph
    static readonly bool PrintDebugInfoToConsole = false;
    static readonly int NumberOfGenerationForTaskCount = 5;

    //static int[] possibleTaskCounts = new[] {30,40,50,60,70,80,100}; //la versione vettorizzata del kernel richiede che n_nodes sia potenza di 2
    static readonly int[] possibleTaskCounts = new[] { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    static int taskCount;
    static readonly float[] possibleShapes = new[] { 0.5f, 1.0f, 2.0f };
    static float shape;
    static readonly int[] possibleOutDegrees = new[] { 1, 2, 3, 4, 5 };
    static int outDegree;
    static readonly float[] possibleComCompRatio = new[] { 0.1f, 0.5f, 1.0f, 5.0f, 10.0f };
    static float comCompRatio;
    static readonly float[] possibleCompCostRange = new[] { 0.1f, 0.5f, 1.0f };
    static float compCostRange;
    static readonly int[] possibleProcessorCount = new[] { 4, 8, 12, 16, 20 };
    static int processorsCount;

    static int height, width;

#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
    static AdjacencyGraph<int, WeightedEdge<int, int>> graph;
    static int[] levelForTask;
    static int[] averageCostForTask;
    static int[][] costForTaskInProcessor;
#pragma warning restore CS8618


    static async Task Main(string[] args)
    {
        for(int i=0; i< NumberOfGenerationForTaskCount; i++)
        {
            ParseInput(args);

            InitGraphWithVetexes();

            SubdivideVertexesInLevels();

            CreateEdges();

            ComputeCostsForEachTaskInProcessors();

            PrintInfo();

            await PrintDataSetOnFile();
        }
    }

    static async Task PrintDataSetOnFile()
    {
        var builder = new StringBuilder();

        builder.AppendLine($"{taskCount} {processorsCount}");

        foreach (var task in graph.Vertices)
        {
            builder.AppendLine(
                $"{averageCostForTask[task]} " +
                $"{string.Join(" ", costForTaskInProcessor[task])} " +
                $"{graph.Edges.Count(e => e.Source == task)} " +
                $"{string.Join(" ", graph.Edges.Where(e => e.Source == task).Select(e => $"{e.Target} {e.Weight}"))} ");
        }

        //builder.AppendLine($"{taskCount}");

        //foreach (var task in graph.Vertices)
        //{
        //    builder.AppendLine(
        //        $"{averageCostForTask[task]} " +
        //        $"{graph.Edges.Count(e => e.Source == task)} " +
        //        $"{string.Join(" ", graph.Edges.Where(e => e.Source == task).Select(e => $"{e.Target} {e.Weight}"))} ");
        //}

        //Info about the generator params used while generating this data set.
        builder.AppendLine();
        builder.AppendLine(
           $"{nameof(taskCount)}: {taskCount} " +
           $"| {nameof(processorsCount)}: {processorsCount} " +
           $"| {nameof(shape)}: {shape} " +
           $"| {nameof(outDegree)}: {outDegree} " +
           $"| {nameof(comCompRatio)}: {comCompRatio} " +
           $"| {nameof(compCostRange)}: {compCostRange} " +
           $"| {nameof(height)}: {height}");

        Directory.CreateDirectory("./data_set");
        await File.WriteAllTextAsync("./data_set/" + Guid.NewGuid().ToString("N") + ".txt", builder.ToString());
    }

    static void ComputeCostsForEachTaskInProcessors()
    {
        costForTaskInProcessor = new int[taskCount][];
        for (int i = 0; i < taskCount; i++)
        {
            costForTaskInProcessor[i] = new int[processorsCount];
        }
        var uniformCosts = new ContinuousUniform(0, 2 * wDag);

        averageCostForTask = new int[taskCount];
        for (int i = 0; i < taskCount; i++)
        {
            averageCostForTask[i] = (int)uniformCosts.Sample();
            for (int j = 0; j < processorsCount; j++)
            {
                costForTaskInProcessor[i][j] = random.Next((int)(averageCostForTask[i] * (1 - compCostRange / 2)), (int)(averageCostForTask[i] * (1 + compCostRange / 2)));
            }
        }


        //print costs info
        if (PrintDebugInfoToConsole)
        {
            for (int i = 0; i < taskCount; i++)
            {
                Console.Write($"{i} ==> ");
                for (int j = 0; j < processorsCount; j++)
                {
                    Console.Write($"{costForTaskInProcessor[i][j]}, ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }
    }

    static void CreateEdges()
    {
        for (int i = 0; i < taskCount - 1; i++)
        {
            var destinations = random.NextInt32Sequence(i + 1, taskCount).Take(outDegree);
            foreach (var node in destinations)
            {
                if (levelForTask[node] > levelForTask[i] && !graph.ContainsEdge(i, node))
                {
                    graph.AddEdge(new WeightedEdge<int, int>(i, node, random.Next(0, 100)));
                }
            }
        }

        //print edges info
        if (PrintDebugInfoToConsole)
        {
            var count = 0;
            for (int i = 0; i < taskCount; i++)
            {
                Console.Write($"{i} ==> ");
                var linkedNodes = graph.Edges.Where(e => e.Source == i).Select(e => e.Target);
                Console.Write(string.Join(", ", linkedNodes));
                Console.WriteLine();

                count += linkedNodes.Count();
            }
            Console.WriteLine();

            var mean = (float)count / taskCount;
            Console.WriteLine($"mean OutDegree espected: {outDegree}, actual {mean}\n");
        }
    }


    static void InitGraphWithVetexes()
    {
        graph = new AdjacencyGraph<int, WeightedEdge<int, int>>(false, taskCount);

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

        if (PrintDebugInfoToConsole)
        {

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
    }

    static void PrintInfo()
    {
            Console.WriteLine();
            Console.WriteLine("Values: {0}: {1} | {2}: {3} | {4}: {5} | {6}: {7} | {8}: {9} | {10}: {11}| {12}: {13}",
                nameof(taskCount), taskCount, nameof(shape), shape, nameof(outDegree), outDegree, nameof(comCompRatio), comCompRatio, nameof(compCostRange), compCostRange, nameof(height), height, nameof(processorsCount), processorsCount);
            Console.WriteLine();
    }

    static void ParseInput(string[] args)
    {
        if (args.Length > 0)
        {
            if (args.Length == 1)
            {
                taskCount = possibleTaskCounts[random.Next(0, possibleTaskCounts.Length)];
                shape = possibleShapes[random.Next(0, possibleShapes.Length)];
                outDegree = possibleOutDegrees[random.Next(0, possibleOutDegrees.Length)];
                comCompRatio = possibleComCompRatio[random.Next(0, possibleComCompRatio.Length)];
                compCostRange = possibleCompCostRange[random.Next(0, possibleCompCostRange.Length)];
                processorsCount = possibleProcessorCount[random.Next(0, possibleProcessorCount.Length)];
            }
            else if (args.Length != 6)
            {
                Console.WriteLine("Usage: TaskGenerator [[random] | [{0} {1} {2} {3} {4} {5}]]", nameof(taskCount), nameof(shape), nameof(outDegree), nameof(comCompRatio), nameof(compCostRange), nameof(processorsCount));
                throw new ArgumentException("Wrong number of arguments", nameof(args));
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
                    processorsCount = int.Parse(args[5]);
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
            do { Console.WriteLine($"{nameof(processorsCount)}: "); } while (!int.TryParse(Console.ReadLine() ?? string.Empty, out processorsCount));
        }
    }
}