# MIL-Team-Complex-Transformers-NLP

**Description**

You have two tasks to deal with. Better do them consiquently, because you are going to use results of the first stage for the second stage. Descriptions, recomendations and requirements related to the particular tasks are listed down below under the corresponding headings. Please read `General requirements` section carefully since these are criterions you will be rated under. 

**General requirements**

The overall requirement is to write clean and readable Python code, which refers to the following points:
1. Write your solution in python-scripts: core components of your solution should be python-scripts, jupyter notebooks are allowded only for experiments; 
2. Use style guides (you can refer to [Google's style guide](https://google.github.io/styleguide/pyguide.html));
3. Pack your solution as a github repository with comprehensive README.md file. We should reach your repo, so you make either public or private one with invitation;
4. Your repo must include:
   - README.md file, requirements for the content see below;
   - Python-scripts that represent your solution;
   - Jupyter notebooks for experiments and trials;
   - requirements.txt file;
   - Other stuff you consider necessary for your solution; 
5. Your README.md must include the following:
   - Obtained results;
   - Description of the methods and approaches you took to get these results, feel free to write down hypothesis and trials that did not succeed;
   - Steps we should proceed to launch your solution and get same results as you did;
   - Everything else you would like to mention for us to better understand your solution;


## Task 1. Implement your own efficient Transformer.
Transformer architecture is known for its superior performance in various NLP domains alongside with quadratic time and memory complexity. In this task you are suggested to write efficient Transformer in terms of time and memory requirements. **NOTE**: you are NOT supposed to write the whole architecture from scratch: highly likely, you are going to find that the only thing you can alter to enhance computational performance is self-attention layer. So, spend your time wisely: rewrite the layers that are needed to be changed, insert them in existing model and reuse pretrained weights. But again, you are free to try whatever you want to accomplish the task. 

For quick start you can refer to [this](https://arxiv.org/abs/2103.13076), [this](https://arxiv.org/abs/2006.16236) and [this](https://arxiv.org/abs/2009.14794). You are not limited to the ideas listed in these papers, your goal is to write faster version of Transformer no matter which approach you are using. 

Note there are existing implementations and you may use them as an inspiration, but you are not allowed to just copy-past somebody's code. In the README.md file while describing your solution you must write down following passages:
1. The overall idea and where did it come from? 
2. What is time and memory complexity of your solution? Prove it with equations.
3. Which parts of the architecture correspond to the pefrormance (time / memory complexity) gains? 
4. How does your solution affect vanilla architecture of the Transformer? 
5. Which benefints and which probable downsides may your solution experience? 

## Task 2. Hit the GLUE!
This task is about finetuning your efficient Transformer on the GLUE tasks. Find two-three tasks among others, which you consider the most demonstrative, and list corresponding performance of your models. You can run your experiments in notebooks. Try to reach as best performance as you can on each of your task, but we do not ask you to do SOTA. Here is the things your report of stage two must include:
1. Description of the chosen tasks: brief overview, metrics and SOTA;
2. Task metrics (efficient Transformer);
3. Reference task metrics (vanilla Transformer);
4. Latency benchmarks;

NOTE: you can choose any architecture of the Transformers family and modify it so it is efficient. Next you measure performance (metrics and latency) of the source version and your optimized version. Your particular task here is to find the way you will be presenting the results so that we can understand performance gains and that is was obtained on the regular basis and not by accident (this refers both to metrics and latency). 


# MIL-Команда-Комплекс-Трансформеры-НЛП

**Описание**

Вам предстоит решить две задачи. Лучше делайте это последовательно, потому что вы собираетесь использовать результаты первого этапа для второго этапа. Описания, рекомендации и требования, относящиеся к конкретным задачам, перечислены ниже под соответствующими заголовками. Пожалуйста, внимательно прочтите раздел "Общие требования", поскольку именно по этим критериям вы будете оцениваться.

**Общие требования**

Общее требование заключается в написании чистого и читаемого кода на Python, который относится к следующим пунктам:
1. Напишите свое решение на python-скриптах: основными компонентами вашего решения должны быть python-скрипты, записные книжки jupyter разрешены только для экспериментов;
2. Используйте руководства по стилю (вы можете обратиться к [руководству Google по стилю](https://google.github.io/styleguide/pyguide.html ));
3. Упакуйте свое решение в виде репозитория github с полным README.md файл. Мы должны связаться с вашим репо, поэтому вы создаете либо публичное, либо частное репо с приглашением;
4. Ваше репо должно включать:
- README.md файл, требования к содержимому см. ниже;
- Python-скрипты, которые представляют ваше решение;
- Записные книжки Jupyter для экспериментов и испытаний;
- requirements.txt файл;
- Другие материалы, которые вы считаете необходимыми для вашего решения;
5. Ваш README.md должно включать в себя следующее:
- Полученные результаты;
- Описание методов и подходов, которые вы использовали для получения этих результатов, не стесняйтесь записывать гипотезы и испытания, которые не увенчались успехом;
- Шаги, которые мы должны предпринять, чтобы запустить ваше решение и получить те же результаты, что и вы;
- Все остальное, что вы хотели бы упомянуть, чтобы мы лучше поняли ваше решение;


## Задание 1. Создайте свой собственный эффективный трансформатор.
Архитектура Transformer известна своей превосходной производительностью в различных областях NLP наряду с квадратичной сложностью по времени и памяти. В этой задаче вам предлагается написать эффективный трансформатор с точки зрения требований ко времени и памяти. ** ПРИМЕЧАНИЕ **: вы не должны писать всю архитектуру с нуля: весьма вероятно, вы обнаружите, что единственное, что вы можете изменить для повышения вычислительной производительности, - это уровень самоконтроля. Итак, потратьте свое время с умом: перепишите слои, которые необходимо изменить, вставьте их в существующую модель и повторно используйте предварительно подготовленные веса. Но опять же, вы вольны пробовать все, что захотите, для выполнения этой задачи.

Для быстрого начала вы можете обратиться к [этому](https://arxiv.org/abs/2103.13076 ), [это](https://arxiv.org/abs/2006.16236 ) и [это](https://arxiv.org/abs/2009.14794 ). Вы не ограничены идеями, перечисленными в этих статьях, ваша цель - написать более быструю версию Transformer, независимо от того, какой подход вы используете.

Обратите внимание, что существуют существующие реализации, и вы можете использовать их в качестве источника вдохновения, но вам не разрешается просто копировать чей-либо код. В самом README.md файл при описании вашего решения вы должны записать следующие отрывки:
1. Общая идея и откуда она взялась?
2. Какова сложность вашего решения по времени и памяти? Докажите это с помощью уравнений.
3. Какие части архитектуры соответствуют выигрышу в производительности (время / сложность памяти)?
4. Как ваше решение влияет на ванильную архитектуру трансформатора?
5. Какие преимущества и какие вероятные недостатки могут быть у вашего решения?

## Задание 2. Попал в КЛЕЙ!
Эта задача посвящена точной настройке вашего эффективного трансформатора для выполнения задач склеивания. Найдите среди других две-три задачи, которые вы считаете наиболее показательными, и перечислите соответствующие показатели ваших моделей. Вы можете проводить свои эксперименты в записных книжках. Постарайтесь достичь максимально возможной производительности при выполнении каждой из ваших задач, но мы не просим вас выполнять SOTA. Вот что должен включать ваш отчет о втором этапе:
1. Описание выбранных задач: краткий обзор, показатели и SOTA;
2. Показатели задачи (эффективный трансформатор);
3. Эталонные показатели задачи (vanilla Transformer);
4. Контрольные показатели задержки;

ПРИМЕЧАНИЕ: вы можете выбрать любую архитектуру семейства Transformers и модифицировать ее таким образом, чтобы она была эффективной. Затем вы измеряете производительность (показатели и задержку) исходной версии и вашей оптимизированной версии. Ваша конкретная задача здесь - найти способ представления результатов, чтобы мы могли понять прирост производительности, который был получен на регулярной основе, а не случайно (это относится как к метрикам, так и к задержке).