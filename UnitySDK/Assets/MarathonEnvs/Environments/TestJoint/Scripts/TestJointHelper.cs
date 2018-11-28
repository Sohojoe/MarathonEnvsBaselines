using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class TestJointHelper : MonoBehaviour {

    [System.Serializable]
    public class MusclePower
    {
        public string Muscle;
        public float Power;
    }

    public List<MusclePower> MusclePowers;

	// Use this for initialization
	void Start () {
		Setup();
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	void Setup () {
		// handle collision overlaps
        IgnoreCollision("torso", new []{"left_upper_arm", "right_upper_arm"});
        IgnoreCollision("butt", new []{"left_thigh", "right_thigh"});

        IgnoreCollision("left_larm", new []{"left_upper_arm"});
        IgnoreCollision("right_upper_arm", new []{"right_larm"});
        IgnoreCollision("left_shin", new []{"left_thigh"});
        IgnoreCollision("right_shin", new []{"right_thigh"});

        IgnoreCollision("right_shin", new []{"right_right_foot"});
        IgnoreCollision("left_shin", new []{"left_left_foot"});


        //
        var joints = GetComponentsInChildren<Joint>().ToList();
        foreach (var joint in joints)
            joint.enablePreprocessing = false;
	}
	void IgnoreCollision(string first, string[] seconds)
    {
        foreach (var second in seconds)
        {
            IgnoreCollision(first, second);
        }
    }
    void IgnoreCollision(string first, string second)
    {
        var rigidbodies = GetComponentsInChildren<Rigidbody>().ToList();
        var rbOne = rigidbodies.FirstOrDefault(x=>x.name.Contains(first))?.GetComponent<Collider>();
        var rbTwo = rigidbodies.FirstOrDefault(x=>x.name.Contains(second))?.GetComponent<Collider>();
        if (rbOne != null && rbTwo != null)
            Physics.IgnoreCollision(rbOne, rbTwo);
    }
}
