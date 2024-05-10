//import org.pytorch.IValue;
import java.util.Arrays;


class JavaTorch {
  static {
    System.loadLibrary("JavaTorch");
  }

  private native float[] inference(String model_path);

  public static void main(String [] args) {
    if (args.length != 1){
      throw new IllegalArgumentException("Give path to model");
    }
    float embedding[] = new JavaTorch().inference(args[0]);
    System.out.println(Arrays.toString(embedding));

  }
}
