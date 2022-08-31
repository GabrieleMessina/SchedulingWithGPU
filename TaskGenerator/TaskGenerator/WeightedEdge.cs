using QuikGraph;

namespace TaskGenerator;

public class WeightedEdge<TVertex, TWeight> : IEdge<TVertex>
{
    public TVertex Source { get; }
    public TVertex Target { get; }
    public TWeight Weight { get; }

    public WeightedEdge(TVertex source, TVertex target, TWeight weight) : base()
    {
        Source = source;
        Target = target;
        Weight = weight;
    }
}