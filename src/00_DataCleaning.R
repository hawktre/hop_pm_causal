# Load in required libraries
library(tidyverse)
library(here)
library(janitor)

#Load in the unclean data
hops <- read.csv(here("data/raw/data_2017_v2.csv"), na.strings = ".") |>
  clean_names()
head(hops)

#Convert date column to actual date type
hops <- hops |>
  mutate(date = mdy(date), month = month(date, label = T), year = year(date))

#correct field names and initial strain names (Strains are V6, Plant type are R6 or non-R6)
hops <- hops |>
  rename(
    "susceptibility_to_v6_strains" = susceptibility_to_r6_strains,
    "susceptibility_to_non_v6_strains" = susceptibility_to_non_r6_strains,
    "plants_sampled" = hill,
    "plants_infected" = w_pm
  ) |>
  mutate(
    initial_strain = case_when(
      initial_strain == "non-R6" ~ "non-V6",
      initial_strain == "R6" ~ "V6",
      T ~ NA
    ),
    cultivar = if_else(
      susceptibility_to_non_v6_strains > 0 |
        is.na(susceptibility_to_non_v6_strains),
      "non-R6",
      "R6"
    )
  )

#Separate disease/treatment columns and wind columns
disease <- hops |>
  select(
    field_id,
    year,
    month,
    date,
    grower,
    centroid_lat,
    centroid_long,
    area_acres,
    cultivar,
    initial_strain,
    plants_sampled,
    plants_infected,
    flag_shoots,
    flag_shoot_incidence,
    contains("pruning"),
    contains("sprays"),
    contains("spray"),
    basal_foliage_removal
  )

wind <- hops |>
  select(
    field_id,
    year,
    month,
    date,
    grower,
    contains("avg"),
    contains("percent")
  )

# Confirm initial_strain where possible. Set to "non-V6" if incidence = 0
disease <- disease |>
  mutate(
    initial_strain = case_when(
      cultivar == "R6" ~ "V6",
      cultivar == "non-R6" &
        is.na(initial_strain) &
        plants_infected == 0 ~ "unknown",
      TRUE ~ initial_strain
    ),
  )

disease |>
  mutate(mildew_incidence = plants_infected / plants_sampled) |>
  arrange(field_id, year, month) |>
  group_by(field_id, year, month) |>
  summarise(N = n(), samples_agee = ) |>
  filter(N > 1)
#Lag the appropriate variables
disease_lagged <- disease |>
  arrange(field_id, year) |>
  group_by(field_id, year) |>
  mutate(
    plants_sampled_prev = lag(plants_sampled, 1),
    plants_infected_prev = lag(plants_infected, 1),
    sprays_prev = lag(monthly_sprays, 1),
    date_prev = lag(date, 1)
  ) |>
  ungroup()
