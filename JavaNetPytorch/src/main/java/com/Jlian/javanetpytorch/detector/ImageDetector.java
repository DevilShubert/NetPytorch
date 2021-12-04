package com.Jlian.javanetpytorch.detector;

/**
    *Detector层：用于传递图像url资源路径或编码后的数据
    * @author JLian
    * @date 2021/11/16 9:57 下午
*/

public interface ImageDetector {
    /**
        * @param imageBase64Code
        * @return java.lang.String
        * @author JLian
        * @date 2021/11/16 9:58 下午
    */
    String getDetectImageByCode(String imageBase64Code) throws InterruptedException;
}
