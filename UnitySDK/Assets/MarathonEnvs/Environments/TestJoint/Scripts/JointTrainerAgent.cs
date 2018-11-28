using System.Collections;
using System.Collections.Generic;
using MLAgents;
using UnityEngine;

public class JointTrainerAgent : Agent {
    
    public int Scale = 10;
    int _subStep;

    float _posDistance; // distance above
    float _negDistance; // distance below
	bool _isStill;
	bool _isOnTarget;
    
    TestJointAgent _jointAgent;
    DiscreteTestJointAgent _discreteJointAgent;

    public override void AgentReset()
	{
        _jointAgent = GetComponent<TestJointAgent>();
        _discreteJointAgent = GetComponent<DiscreteTestJointAgent>();
        _subStep = 0;
		_negDistance = 0f;
		_posDistance = 0f;
		_isStill = false;
		_isOnTarget = false;
    }

    public override void CollectObservations()
    {
        var muscles = _jointAgent?.Muscles ?? _discreteJointAgent.Muscles;
        foreach (var m in muscles)
        {
            var diff = m.ObsNormalizedAngleX - m.TargetNormalizedAngleX;
            _posDistance = Mathf.Clamp(diff,0f,1f);
            _negDistance = Mathf.Clamp(-diff,0f,1f);
			_isStill = m.ObsRotationVelocity > -1e-2 && m.ObsRotationVelocity < 1e-2;
			_isOnTarget = diff > -1e-2 && diff < 1e-2;
            AddVectorObs(_posDistance);
            AddVectorObs(_negDistance);
            AddVectorObs(m.TargetAngularVelocityX);
            AddVectorObs(m.ObsNormalizedAngleX);
            AddVectorObs(m.TargetNormalizedAngleX);
            AddVectorObs(m.ObsRotationVelocity);
            AddVectorObs(_isStill);
            AddVectorObs(_isOnTarget);
        }
    }
	public override void AgentAction(float[] vectorAction, string textAction)
	{
		int i = 0;
        var muscles = _jointAgent?.Muscles ?? _discreteJointAgent.Muscles;
        foreach (var m in muscles)
        {
            m.TargetNormalizedAngleX = vectorAction[i++];
        }
	}

    // internal void Terminate(float cumulativeReward)
	// {
	// 	if (this.IsDone())
	// 		return;
	// 	var maxReward = 1000f;
	// 	var agentReward = cumulativeReward;
	// 	agentReward = Mathf.Clamp(agentReward, 0f, maxReward);
	// 	var adverseralReward = maxReward - agentReward;
	// 	AddReward(adverseralReward);
	// 	Done();
	// }



    internal void Terminate()
    {
        if (this.IsDone())
			return;
	    Done();
    }
    internal void ChildStep(float reward)
    {
        var adverseralReward = - reward;
        AddReward(adverseralReward);
    }
}