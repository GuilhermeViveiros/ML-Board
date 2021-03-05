O que verifico até agora:

Implementei Vanilla Policy Gradient, Deep Q Network (DQN) e Advantage Actor Critic (A2C)

Policy gradient e A2C são mais rápidos em comparação com DQN, porque DQN é off-policy e utiliza replay buffer no processo.
No entanto PG e A2C (com apenas um worker) necessitam de muitos mais steps para uma boa otimização, mais ao menos 1000 steps. DQN em apenas 150 steps já tinge um resultado ótimo.

PG percebe-se que demore mais porque este não faz diferenças entre a composição de rewards, ou seja caso tenho uma má reward pode a repetir se em média não for muito má, isto pode trazer problemas visto que este método faz os seus updates com base em vários retornos, em casos normais complica-se porque não temos estas possibilidades, temos que fazer um processo incremental através de TD.

DQN e A2C fazem isto. 
DQN utiliza replay buffer por isso faz uma espécie de planning RL, é por isso que é mais lento e também demora menos steps a convergir. Para além disso não implementei uma target network para o td error no A2C o que torna o algoritmo mais instável, com mais stationarity. **Também só utilizo um worker, com vários workers a processarem simultaneamente este problema pode ser resolvido mais rápido. É algo que quero testar.** 


A2C com dois optimizers para cada network melhorou o performance da tarefa extrordinariamente. Para além de ser mais rápido e só utilizar CPU, resolve o problema mais rápido que DQN.


Passos no A2C -> target value network and more workers

Implementar PPO e ver PacMan

