import java.util.Arrays;

class DummyLightCurve {
  public double[] time;
  public double[] mag;
  public double[] err;

  public DummyLightCurve(int N) {
    time = new double[N];
    mag = new double[N];
    err = new double[N];
    for (int i = 0; i < N; i++) {
      time[i] = 1000.0 * ((double)i / ((double)N - 1));
      mag[i] = Math.sin(2.0 * Math.PI * time[i]);
      err[i] = 0.1;
    }
  }
}

class JavaTorch {
  private native float[] inference(String model_path, DummyLightCurve lc,
                                   boolean is_gpu);

  public static void main(String[] args) {
    if (args.length != 2) {
      throw new IllegalArgumentException(
          "Arg1 (str): path to model, Arg2 (bool) use_GPU");
    }
    DummyLightCurve lc = new DummyLightCurve(100);
    float embedding[] =
        new JavaTorch().inference(args[0], lc, Boolean.parseBoolean(args[1]));
    System.out.println(Arrays.toString(embedding));
  }
  static { System.loadLibrary("JavaTorch"); }
}
