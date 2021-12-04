package com.Jlian.javanetpytorch.constant;

/**
    * 试图层常量定义
    * @author JLian
    * @date 2021/11/16 10:16 下午
*/

public class ViewModelConstants {
    /**
     * index.html
     */
    public static final String DETECT = "site/index";

    /**
     * detectOut.html
     */
    public static final String DETECT_OUT = "site/detectOut";

    /**
     * image on detectOut
     */
    public static final String DETECT_OUT_IMAGE = "img";

    /**
     * notice on detectOut
     */
    public static final String DETECT_OUT_NOTICE = "notice";

    /**
     * 理论上最大图片大小为10MB
     */
    public static int MAX_FILE_SIZE = 1024 * 1024 * 10;

    /**
     * 访问pytorch层的url
     */

    public static String HTTP_URL = "http://127.0.0.1:5000/detect/imageDetect";
}
