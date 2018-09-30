public class Vector
{
    public final int length;
    public final double[] data;

    public Vector(int length)
    {
        this.length = length;
        data = new double[length];
    }

    public Vector(double[] data)
    {
        length = data.length;
        this.data = new double[length];
        for (int l = 0; l < length; l++)
                this.data[l] = data[l];
    }

    private Vector(Vector other)
    {
        this(other.data);
    }

    public double mean()
    {
        double sum = 0;
        for(int i = 0; i < length; i++)
            sum += data[i];
        return sum / (double)length;
    }

    public Vector abs()
    {
        Vector result = new Vector(this);
        for (int i = 0; i < length; i++)
            if (result.data[i] < 0)
                result.data[i] = -result.data[i];
        return result;
    }

    public double max()
    {
        double max = 0;
        for (int i = 0; i < length; i++)
            if (data[i] > max)
                max = data[i];
        return max;
    }

    public double norm()
    {
        double norm = 0;
        for (int i = 0; i < length; i++)
            norm += data[i] * data[i];
        return Math.sqrt(norm);
    }

    public Vector plus(double scalar)
    {
        Vector result = new Vector(this);
        for (int i = 0; i < length; i++)
            result.data[i] += scalar;
        return result;
    }

    public Vector plus(Vector other)
    {
        Vector result = new Vector(this);
        for(int i = 0; i < length; i++)
            result.data[i] += other.data[i];
        return result;
    }

    public Vector minus(Vector other)
    {
        Vector result = new Vector(this);
        for(int i = 0; i < length; i++)
            result.data[i] -= other.data[i];
        return result;
    }

    public Vector times(double scalar)
    {
        Vector result = new Vector(this);
        for(int i = 0; i < length; i++)
            result.data[i] *= scalar;
        return result;
    }

    public double times(Vector other)
    {
        double result = 0;
        for (int i = 0; i < length; i++)
            result += this.data[i] * other.data[i];
        return result;
    }

    public double inf_distance(Vector other)
    {
        double maxDif = 0;
        for (int i = 0; i < length; i++)
            if(Math.abs(this.data[i] - other.data[i]) > maxDif)
                maxDif = Math.abs(this.data[i] - other.data[i]);
        return maxDif;
    }

    public void print()
    {
        for (int i = 0; i < length; i++){
            System.out.print(data[i]);
            System.out.print(" ");
        }
        System.out.println();
    }
}
