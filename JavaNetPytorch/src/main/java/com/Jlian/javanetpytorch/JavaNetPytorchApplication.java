package com.Jlian.javanetpytorch;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication
@EnableCaching
public class JavaNetPytorchApplication {

    public static void main(String[] args) {
        SpringApplication.run(JavaNetPytorchApplication.class, args);
    }

}
