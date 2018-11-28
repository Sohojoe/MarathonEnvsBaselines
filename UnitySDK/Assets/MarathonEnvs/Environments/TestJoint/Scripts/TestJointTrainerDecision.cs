using System.Collections.Generic;
using System.Linq;
using MLAgents;
using UnityEngine;

public class TestJointTrainerDecision : MonoBehaviour, Decision
{
    Brain _brain;

    public float[] Actions;
    public bool ApplyRandomActions;
    public bool ApplySeededActions;
    public int Seed = 33;
    public int SeededActionsCount = 100;
    public List<float> SeededActions;
    public float AverageReward;

    int _lastSeed;
    List<float> _rewards;
    const int _rewardMax = 500;

    void Start()
    {
        _brain = GetComponent<Brain>();
        Actions = Enumerable.Repeat(0f, _brain.brainParameters.vectorActionSize[0]).ToArray();
        SetSeededActions();
    }
    void Update()
    {
        if (Seed != _lastSeed)
            SetSeededActions();
    }

    void SetSeededActions()
    {
        Random.State oldState = Random.state;
        Random.InitState(Seed);
        SeededActions = Enumerable
            .Repeat(0, SeededActionsCount)
            .Select(x=> Random.value*2-1)
            .ToList();
        Random.state = oldState;
        _lastSeed = Seed;
        _rewards = new List<float>(_rewardMax);
        AverageReward = 0f;
    }

    public void SetReward(float cumulativeReward)
    {
        if (_rewards.Count >= _rewardMax)
            _rewards.RemoveAt(0);
        _rewards.Add(cumulativeReward);
        AverageReward = _rewards.Average();
    }


    public float[] Decide(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        float posDistance = vectorObs[0];
        float negDistance = vectorObs[1];
        float targetAngularVelocityX = vectorObs[2];
        float obsNormalizedAngleX = vectorObs[3];
        float targetNormalizedAngleX = vectorObs[4];
        float obsRotationVelocity = vectorObs[5];
        

        if (posDistance > 0.001f) {
            if (targetAngularVelocityX < 0)
                targetAngularVelocityX = 0;
            else
                targetAngularVelocityX = 0.1f;
        }
        else if (negDistance > 0.001f){
            if (targetAngularVelocityX > 0)
                targetAngularVelocityX = 0;
            else
                targetAngularVelocityX = 0.05f;
        }
        Actions[0] = targetAngularVelocityX;

        if (ApplySeededActions)
        {
            for (int i = 0; i < Actions.Length; i++)
                Actions[i] = SeededActions[Random.Range(0, SeededActions.Count)];
        }
        else if (ApplyRandomActions)
        {
            for (int i = 0; i < Actions.Length; i++)
                Actions[i] = Random.value * 2 - 1;
        }

        return Actions;
    }

    public List<float> MakeMemory(
        List<float> vectorObs,
        List<Texture2D> visualObs,
        float reward,
        bool done,
        List<float> memory)
    {
        return new List<float>();
    }
}