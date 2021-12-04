package com.Jlian.javanetpytorch.constant;

public class RedisConstant {
    /**
     * max thread queueLength
     */
    public static final long QUEUE_SIZE = 3;

    // 大并发来临时的queue，用list模拟队列
    public static final String IMAGE_QUEUE_LIST = "imageQueueList";

    // 用作缓存的Hash表，key为每个图片的UUID，value为返回的图片
    public static final String CACHE_HASH = "cacheHash";

    // 超过最大的栈返回的提醒
    public static final String  QUEUE_OVERFLOW_NOTICE = "System busy, Please try again later";

    // 队列列表的key
    public static final String  QUEUE_LIST_KEY = "queueListKey";

    // 队列列表的value
    public static final String  QUEUE_LIST_VALUE = "queueListValue";
}
