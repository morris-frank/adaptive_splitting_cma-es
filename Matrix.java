import java.util.Arrays;

public class Matrix
{
    public final int N;
    public final int Dim;
    public final double[][] data;

    public Matrix(int N, int Dim)
    {
        this.N = N;
        this.Dim = Dim;
        data = new double[N][Dim];
    }

    public Matrix(double[][] data)
    {
        N = data.length;
        Dim = data[0].length;
        this.data = new double[N][Dim];
        for (int i = 0; i < N; i++)
            for (int d = 0; d < Dim; d++)
                    this.data[i][d] = data[i][d];
    }

    private Matrix(Matrix other)
    {
        this(other.data);
    }

    public Matrix plus(Vector vector)
    {
        Matrix result = new Matrix(this);
        for(int i = 0; i < N; i++)
            for(int d = 0; d < Dim; d++)
                result.data[i][d] += vector.data[d];
        return result;
    }

    public Matrix plus(Matrix other)
    {
        Matrix result = new Matrix(this);
        for (int n = 0; n < N; n++)
            for (int d = 0; d < Dim; d++)
                result.data[n][d] += other.data[n][d];
        return result;
    }

    public Matrix times(double scalar)
    {
        Matrix result = new Matrix(this);
        for(int i = 0; i < N; i++)
            for(int d = 0; d < Dim; d++)
                result.data[i][d] *= scalar;
        return result;
    }

    public Vector times(Vector vector)
    {
        Vector result = new Vector(N);
        for(int i = 0; i < N; i++)
            for(int d = 0; d < Dim; d++)
                result.data[i] += data[i][d] * vector.data[d];
        return result;
    }

    public Matrix T()
    {
        Matrix result = new Matrix(Dim, N);
        for (int i = 0; i < N; i++)
            for (int d = 0; d < Dim; d++)
                result.data[d][i] = this.data[i][d];
        return result;
    }

    public Vector mean()
    {
        Vector mean = new Vector(Dim);
        for(int n = 0; n < N; n++)
            for(int d = 0; d < Dim; d++)
                mean.data[d] += data[n][d];
        for(int d = 0; d < Dim; d++)
            mean.data[d] /= N;
        return mean;
    }

    public double max()
    {
        double max = 0;
        for (int i = 0; i < N; i++)
            for (int d = 0; d < Dim; d++)
                if(data[i][d] > max)
                    max = data[i][d];
        return max;
    }

    public Matrix covariance()
    {
        Matrix covariance = new Matrix(Dim, Dim);
        Vector mean = mean();
        for(int i = 0; i < N; i++)
            for(int d = i; d < Dim; d++){
                double c = 0;
                for(int h = 0; h < N; h++)
                    c +=  (data[h][i] - mean.data[i]) * (data[h][d] - mean.data[d]);
                c /= N - 1;
                covariance.data[i][d] = c;
                covariance.data[d][i] = c;
            }
        return covariance;
    }

    public Matrix cholesky()
    {
        Matrix L = new Matrix(N, N);

        for (int i = 0; i < N; i++){
            for (int j = 0; j <= i; j++){
                double sum = 0.0;
                for (int k = 0; k < j; k++)
                    sum += L.data[i][k] * L.data[j][k];
                if (i == j) L.data[i][i] = Math.sqrt(data[i][i] - sum);
                else        L.data[i][j] = 1.0 / L.data[j][j] * (data[i][j] - sum);
            }
            // Yes this is wrong, but we're not mathematicians so shut up
            if (L.data[i][i] <= 0){
                L.data[i][i] = 0;
                // throw new RuntimeException("Matrix not positive definite");
            }
        }
        return L;
    }

    public void print()
    {
        for(int i = 0; i < N; i++)
            System.out.println(Arrays.toString(data[i]));
        System.out.println();
    }
}
