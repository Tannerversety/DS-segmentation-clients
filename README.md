# DS-segmentation-clients
Nous cherchons à segmenter des clients en fonction de leurs comportements d'achat. Pour cela nous utilisons une base de donnée contenant plusieurs milliers de transactions. Nous construisons les variables adéquates fonction de la date d'achat/montant/quantité/prix. 
Nous utilisons deux méthodes pour générer des variables d'interets. Une méthode directe, une simple fonction des variables d'entrées et une méthode indirecte 
reposant sur l'utilisation d'un filtre collaboratif implicite. Les variables (vecteurs features) émergent du comportement seul des utilisateurs. Ces
dernières sont donc difficilement interprétables. Etant donné les items contenus dans la base de donnée sont très analogues, cette méthode n'a 
pas porté les fruits escomptés.

Une fois les variables générées, nous effectuons une segmentation avec des algorithmes de clustering, nous calibrons le nombre optimal de cluster
avec une métrique de type "indice de rand ajusté". Nous interprétons ensuite qualitativement la nature des clusteurs en visualisant la moyenne des variables sur
chaque clusteur. 

Enfin, nous quantifions la performance de prédiction de notre travail sur jeu de test en utilisant un algorithme de type forêt aléatoire (random forrest).
On entraine l'algorithme à prédire le bon clusteur établissant différents scenarii.
