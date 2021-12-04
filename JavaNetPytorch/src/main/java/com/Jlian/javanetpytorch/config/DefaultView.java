package com.Jlian.javanetpytorch.config;

import com.Jlian.javanetpytorch.constant.ViewModelConstants;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.web.servlet.config.annotation.ViewControllerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class DefaultView implements WebMvcConfigurer {

    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        registry.addViewController("/").setViewName(ViewModelConstants.DETECT);
        registry.addViewController("/detect").setViewName(ViewModelConstants.DETECT);
        registry.setOrder(Ordered.HIGHEST_PRECEDENCE);
    }
}

