library(dplyr)
library(ggplot2)
library(DBI)
library(RPostgres)
library(magrittr)

pl_lkup = Lahman::Master %>% dplyr::select(playerID, retroID, bbrefID, nameFirst, nameLast)
pl_lkup$nameAbbv = paste(stringr::str_sub(pl_lkup$nameFirst, 1, 1), pl_lkup$nameLast, sep='.')


get_data = function(year_min=2016, year_max=2016, group_inn=FALSE) {
  woba_df = data.frame(
    event_cd = c(2, 3, 14, 15, 16, 18, 20, 21, 22, 23),
    woba_pts = c(0, 0, rep(0.7, 4), 0.9, 1.25, 1.6, 2)
  )

  # conn here is a connection to a postgres database holding retrosheet event data
  b = dbGetQuery(conn,
                 paste0("select * from event where ",
                        "year_id>=", year_min, " and year_id<=", year_max)
                 )

  b %<>% mutate(i2=event_cd==2,
                i3=event_cd==3,
                i14=event_cd %in% c(14:16, 18),
                i20=event_cd==20,
                i21=event_cd==21,
                i22=event_cd==22,
                i23=event_cd==23,
                is_pa= (ab_fl | i14 | sf_fl | sh_fl ))


  b %<>% left_join(woba_df, by="event_cd")
  if (group_inn) {
    groupers = quos(game_id, bat_home_id, inn_ct)
  } else {
    groupers = quos(game_id, bat_home_id)
  }
  b %>%
    filter(inn_ct<=8) %>%
    group_by(!!!groupers) %>%
    summarise(outs=sum(event_outs_ct),
              runs=sum(event_runs_ct), pa=sum(is_pa),
              woba=sum(woba_pts, na.rm=TRUE) / pa,
              i2=sum(i2), i3=sum(i3), i14=sum(i14), i20=sum(i20),
              i21=sum(i21), i22=sum(i22), i23=sum(i23), max_inn=max(inn_ct)) %>%
    ungroup() %>%
    group_by(game_id) %>%
    filter(max_inn >= 8) %>%
    ungroup() %>%
    mutate(k=paste(i2, i3, i14, i20, i21, i22, i23, sep='_'),
           kx=paste(i14, i20, i21, i22, i23, sep='_'),
           skx=i14+i20+i21+i22+i23)
}

conditional_dist_plot = function(seq_df, grouper_, ...) {
  grouper = enquo(grouper_)
  groupers = enquos(...)
  
  seq_df %>% 
    filter(pa>=3) %>% 
    group_by(...) %>% 
    summarise(nr=n()) %>% 
    group_by(!!grouper) %>% 
    mutate(n=sum(nr)) %>% 
    ungroup() %>% 
    mutate(z=nr/n) %>%
    ggplot(aes(x=runs, y=z)) + 
    geom_bar(stat='identity', color='steelblue', fill='steelblue') + 
    facet_wrap(vars(!!grouper), nrow=3, labeller = label_both) + 
    theme_minimal(base_size = 16) + 
    labs(x="Runs", y="Probability")
  
  }

variance_decompose = function(seq_df, by_var_) {
  by_var = enquo(by_var_)
  var_df = seq_df %>% 
    filter(pa>=3, pa<=14) %>% 
    group_by(!!by_var) %>% 
    summarise(m=mean(runs, na.rm=T), v=var(runs, na.rm=T), n=n()) %>% 
    ungroup() %>%
    mutate(z=n/sum(n))
  
  tmp = var_df %>% mutate(w=m*z, w2=m*m*z)
  ev = with(var_df, sum(n * v, na.rm=T) / sum(n, na.rm=T))
  ve = sum(tmp$w2, na.rm=T) - sum(tmp$w, na.rm=T)**2
  list(ev=ev, ve=ve)
}