## TManual analysis example
## N. Mizumoto

#------------------------------------------------------------------------------#
# The original data from Mizumoto et al. 2020 Am Nat DOI:10.1086/711020
# Analyzed using TManual
#------------------------------------------------------------------------------#

library(data.table)
library(stringr)

library(ggplot2)
library(viridis)

library(extrafont)
#font_import(pattern="PT")
loadfonts()

df = fread("data/df_summary.csv", header=T)
df$Species = str_split(df$id, "-", simplify = T)[,1]
df$Colony = str_split(df$id, "-", simplify = T)[,2]
df$Trial = str_split(df$id, "-", simplify = T)[,3]

ggplot(df, aes(x=serial, y=tunnel_length_total, col=Species))+
  scale_color_viridis(discrete=T, alpha=0.5, option="A") +
  stat_smooth() +
  geom_point() +
  theme_bw() + 
  theme(legend.justification = c(-0.1,1.1),
        legend.position = c(0,1), aspect.ratio = 1/2)+
  xlab("Hours") +
  ylab("Total tunnel length (mm)")
ggsave(paste0("img/Length.pdf"),
       width=7, height = 4, family="PT Sans")

df_prop = data.frame(
  df[ ,c("Species", "serial")],
  tunnel_id = rep(c("1st","2nd","3rd","4more"), each=dim(df)[1]),
  tunnel_prop =
    c(df$tunnel_length_1st, df$tunnel_length_2nd,                  
      df$tunnel_length_3rd, df$tunnel_length_4more)/df$tunnel_length_total
  )

ggplot(df_prop, aes(x=serial, y=tunnel_prop, col=tunnel_id))+
  stat_smooth()+
  geom_point(alpha=0.2)+
  scale_color_viridis(begin=0, end=0.8, discrete=T) +
  theme_bw() + 
  theme(legend.justification = c(-0.1,1.1),
        legend.position = c(0,1), aspect.ratio = 1/1.25)+
  xlab("Hours") +
  ylab("Proportion of tunnels")+
  facet_grid(~Species)
ggsave(paste0("img/Structures.pdf"),
       width=10, height = 3, family="PT Sans")

