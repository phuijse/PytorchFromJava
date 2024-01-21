class JavaTorch {
  static {
    System.loadLibrary("JavaTorch");
  }

  private native int inference(String model_path);

  public static void main(String [] args) {
    if (args.length != 1){
      throw new IllegalArgumentException("Given path to model");
    }
    int inferred_class = new JavaTorch().inference(args[0]);
    System.out.println(inferred_class);

  }
}
