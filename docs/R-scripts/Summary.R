Summary <- function (data,
                     lev = NULL,
                     model = NULL) {
  out <- c(sqrt(mean((data$pred - data$obs)^2,na.rm = TRUE)),
           ((sqrt(mean((data$pred - data$obs)^2,na.rm = TRUE)))/mean(data$obs,na.rm = TRUE))*100,
           mean((data$pred - data$obs)^2,na.rm = TRUE),
           cor(data$pred, data$obs),
           (cor(data$pred, data$obs)^2),
           mad(data$pred - data$obs, na.rm = TRUE),
           mean(abs(data$pred - data$obs), na.rm = TRUE))
  names(out) <- c("RMSE", "RMSE%", "MSE", "r", "Rsq", "MAD", "MAE")
  out
}
